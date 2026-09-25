# Mypy will not try inferring the types of any 3rd party libraries installed.
# mypy: ignore-errors

import bisect
import collections
import concurrent.futures
import dataclasses
import io
import itertools
import logging
import mmap
import os
import secrets
import shutil
import sys
import threading
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec
import fsspec.asyn
from fsspec.core import url_to_fs

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from torch import Tensor
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._extension import StreamTransformExtension
from torch.distributed.checkpoint.filesystem import (
    FileSystemBase,
    FileSystemReader,
    FileSystemWriter,
    SerializationFormat,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
)
from torch.futures import Future
from torch.serialization import _load, _open_zipfile_reader


if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


__all__ = [
    "FsspecWriter",
    "FsspecReader",
]

logger = logging.getLogger(__name__)

_SHM_DIR = "/dev/shm"
# How long a rank waits for an item another rank fetches for it before
# fetching the item itself.
_SHARE_TIMEOUT_S = 600.0
# How much later than its fetch a handed-over batch reaches the rank that owns
# it: the helper opening a file it has not read yet, writing the batch to
# /dev/shm, and the owner polling for and mapping it.
_HANDOVER_S = 0.1
_HANDOVER_BYTES_PER_S = 2e9
# At most this many files are opened ahead of the batches that read them.
_WARM_FILES = 16


class FileSystem(FileSystemBase):
    def __init__(self) -> None:
        self.fs: AbstractFileSystem | None = None

    @contextmanager
    def create_stream(
        self, path: str | os.PathLike, mode: str
    ) -> Generator[io.IOBase, None, None]:
        if self.fs is None:
            raise AssertionError("fs should not be None")
        path = os.fspath(path)

        # fsspec does not support concurrent transactions, and not all
        # AbstractFileSystem have working rollback implementations, so
        # just manually delete the file if necessary on errors.
        with self.fs.open(path, mode) as stream:
            try:
                yield stream
            except:
                if any(ch in mode for ch in "w+a"):  # cleanup file if not read-only
                    try:
                        self.rm_file(path)
                    except:  # noqa: E722
                        pass
                raise

    def concat_path(self, path: str | os.PathLike, suffix: str) -> str | os.PathLike:
        return os.path.join(path, suffix)

    def init_path(self, path: str | os.PathLike, **kwargs) -> str | os.PathLike:
        self.fs, _ = url_to_fs(path, **kwargs)
        return path

    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None:
        self.fs.rename(path, new_path)

    def mkdir(self, path: str | os.PathLike) -> None:
        self.fs.makedirs(path, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        if isinstance(checkpoint_id, Path):
            return False

        try:
            url_to_fs(checkpoint_id)
        except ValueError:
            return False

        return True

    def exists(self, path: str | os.PathLike) -> bool:
        return self.fs.exists(path)

    def rm_file(self, path: str | os.PathLike) -> None:
        self.fs.rm(path)

    def ls(self, path: str | os.PathLike) -> list[str]:
        # setting detail to False explicitly to keep the list[str] return type,
        # instead of the list[Dict] return type when detail=True
        return self.fs.ls(path, detail=False)


# TODO: add the dcp.async_save mixin
class FsspecWriter(FileSystemWriter):
    """
    Basic implementation of StorageWriter using fsspec.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: str | os.PathLike,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        overwrite: bool = True,
        _extensions: Sequence[StreamTransformExtension] | None = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
        **kwargs,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving them. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__(
            path,
            single_file_per_rank,
            sync_files,
            thread_count,
            per_thread_copy_ahead,
            overwrite=overwrite,
            _extensions=_extensions,
            serialization_format=serialization_format,
        )
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


def _destinations_disjoint(targets: list[Tensor]) -> bool:
    """Whether every destination occupies its own bytes.

    Copies into disjoint memory can run concurrently no matter what the planner
    does. Overlap means two items would race, which happens when a planner
    resolves several items onto one staging buffer, and it is also the normal
    case when items narrow into different regions of the same tensor.

    Non-contiguous tensors are rejected rather than analyzed, which keeps
    ``[data_ptr, nbytes)`` an exact extent instead of a bound, so this never
    reports disjoint for targets that actually share bytes.
    """
    spans = []
    for t in targets:
        if not t.is_contiguous():
            return False
        start = t.data_ptr()
        spans.append((start, start + t.numel() * t.element_size()))
    spans.sort()
    return all(end <= nxt for (_, end), (nxt, _) in itertools.pairwise(spans))


def _shm_host_id() -> tuple[str, int] | None:
    """Identify the /dev/shm this process sees, and its free bytes.

    Ranks reporting the same id can hand each other files through it.
    """
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            boot_id = f.read().strip()
        st = os.stat(_SHM_DIR)
        free = shutil.disk_usage(_SHM_DIR).free
    except OSError:
        return None
    return f"{boot_id}:{st.st_dev}:{st.st_ino}", free


_ItemKey = tuple[str, int, int]


@dataclasses.dataclass
class _SharedReads:
    """How the ranks on one host split up their reads.

    An item several of them read is fetched by one owner, which hands it to the
    others as hard links of one /dev/shm file named ``{prefix}{id}.{reader}``.
    Each rank also claims its batches in order with exclusive files named
    ``{prefix}c{rank}.{batch}``. A rank that is out of batches claims the last
    unclaimed batch of the rank with the most bytes left, fetches it and hands
    it over as ``{prefix}h{rank}.{batch}``.
    """

    prefix: str
    rank: int
    send: dict[_ItemKey, tuple[int, list[int]]]
    recv: dict[_ItemKey, int]
    # The ranges of every batch of every rank on the host, in fetch order.
    batches: dict[int, list[list[_ItemKey]]]
    n_shared: int


def _create_excl(path: str) -> bool:
    """Create an empty file at ``path``, or return False if one exists."""
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        return False
    return True


def _publish(tmp: str, paths: list[str], chunks: list) -> None:
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            with open(fd, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
        except OSError:
            # A short file sends the readers back to storage without waiting.
            os.truncate(tmp, 0)
            raise
        finally:
            for path in paths:
                os.link(tmp, path)
    finally:
        os.unlink(tmp)


def _receive_shared(
    path: str, length: int, deadline: float, stop: threading.Event
) -> mmap.mmap | None:
    """Map the file another rank publishes at ``path``, or None to fall back."""
    delay = 0.001
    while True:
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            break
        except FileNotFoundError:
            if time.monotonic() > deadline or stop.is_set():
                return None
            time.sleep(delay)
            delay = min(2 * delay, 0.01)
    try:
        if length == 0 or os.fstat(fd).st_size != length:
            return None
        flags = mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0)
        return mmap.mmap(fd, length, flags=flags, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
        os.unlink(path)


def _leave_host(share: _SharedReads) -> None:
    """Mark this rank done with the host's files; the last rank removes them."""
    p = share.prefix
    ranks = list(share.batches)
    try:
        _create_excl(f"{p}x{share.rank}")
        if not all(os.path.exists(f"{p}x{r}") for r in ranks):
            return
        # Anything still here was handed over after its reader gave up on it.
        names = [f"{p}{n}.{r}" for n in range(share.n_shared) for r in ranks]
        for r, table in share.batches.items():
            for i in range(len(table)):
                names += [f"{p}c{r}.{i}", f"{p}h{r}.{i}"]
        names += [f"{p}x{r}" for r in ranks]
        for name in names:
            with suppress(FileNotFoundError):
                os.unlink(name)
    except OSError as e:
        logger.warning("Could not clean up %s*: %s", p, e)


class _SegmentReader(io.RawIOBase):
    """Seekable stream over byte segments without joining them.

    ``torch.load`` reads through ``readinto``, so the bytes are copied once,
    straight from the fetched buffers into tensor storage.
    """

    def __init__(self, segments: list[memoryview]) -> None:
        super().__init__()
        self._segs = segments
        self._starts = list(itertools.accumulate(map(len, segments), initial=0))
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            pos += self._pos
        elif whence == io.SEEK_END:
            pos += self._starts[-1]
        self._pos = pos
        return pos

    def readinto(self, b) -> int:
        out = memoryview(b).cast("B")
        n = 0
        while n < len(out) and self._pos < self._starts[-1]:
            i = bisect.bisect_right(self._starts, self._pos) - 1
            off = self._pos - self._starts[i]
            k = min(len(out) - n, len(self._segs[i]) - off)
            out[n : n + k] = self._segs[i][off : off + k]
            n += k
            self._pos += k
        return n


def _load_aliased(buf: memoryview, layouts: dict[tuple[bytes, int], tuple]) -> Tensor:
    """torch.load one DCP tensor record, aliasing ``buf`` if the byte order matches.

    Tensors with the same dtype, shape and stride have byte-identical pickles, so
    ``layouts`` lets the weights_only unpickler run once per layout instead of
    once per tensor. It holds the GIL for ~0.3 ms per call, and the event loop
    fetching the next batch waits for it.
    """
    with _open_zipfile_reader(_SegmentReader([buf])) as zf:
        if not (
            zf.has_record("byteorder")
            and zf.get_record("byteorder") == sys.byteorder.encode()
        ):
            return _load(zf, "cpu", _weights_only_unpickler, encoding="utf-8")
        key = None
        if zf.has_record("data/0"):
            start = zf.get_record_offset("data/0")
            key = (zf.get_record("data.pkl"), zf.get_record_size("data/0"))
            if (layout := layouts.get(key)) is not None:
                dtype, size, stride, offset = layout
                count = key[1] // dtype.itemsize
                data = torch.frombuffer(buf, dtype=dtype, count=count, offset=start)
                return data.as_strided(size, stride, offset)
        storage = torch.frombuffer(buf, dtype=torch.uint8).untyped_storage()
        tensor = _load(
            zf,
            "cpu",
            _weights_only_unpickler,
            overall_storage=storage,
            encoding="utf-8",
        )
    if (
        key is not None
        and key[1] > 0
        and type(tensor) is torch.Tensor
        and tensor.layout == torch.strided
        and not (tensor.requires_grad or tensor.is_conj() or tensor.is_neg())
        and not tensor.is_quantized
        and tensor.untyped_storage().data_ptr() == storage.data_ptr() + start
        and tensor.untyped_storage().nbytes() == key[1]
    ):
        layout = (tensor.dtype, tensor.shape, tensor.stride(), tensor.storage_offset())
        layouts[key] = layout
    return tensor


class FsspecReader(FileSystemReader):
    def __init__(
        self,
        path: str | os.PathLike,
        *,
        max_batch_size: int = 1024,
        max_batch_bytes: int = 256 * 1024 * 1024,
        merge_item_bytes: int = 1024 * 1024,
        range_bytes: int = 16 * 1024 * 1024,
        cpu_workers: int | None = None,
        share_reads: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the FsspecReader pointing to `path`.

        Args:
            path: directory or URL where the checkpoint will be read from.
            max_batch_size: Maximum number of ranges per batched cat_ranges call.
                Defaults to 1024, so batches are normally bounded by bytes. Small
                items that cannot be merged, such as per-parameter optimizer
                steps spread over every rank's file, then share one call instead
                of paying for opening their files in each of several calls.
            max_batch_bytes: Maximum cumulative byte size requested per batched
                cat_ranges call. Defaults to 256 MiB. This caps one request, not
                resident memory: the next two batches are fetched while the current
                one is still being decoded and copied, so expect a small multiple of
                this to be live at peak.
            merge_item_bytes: Contiguous items no larger than this are merged
                into one range, since each range has a fixed cost that dominates
                for small items. Larger items are read as their own range.
                Defaults to 1 MiB.
            range_bytes: Maximum size of a merged range. Defaults to 16 MiB.
            cpu_workers: Number of worker threads for parallel CPU deserialization.
                Defaults to min(4, max(1, cpu_count // local_world_size)).
            share_reads: Let the ranks on one host share their reads through
                /dev/shm. Items larger than ``merge_item_bytes`` that several of
                them read (e.g. tensors replicated across tensor parallel ranks)
                are fetched once, and a rank that finishes its own batches early
                fetches the last batches of slower ranks for them. Defaults to
                True.
            **kwargs: Additional storage options passed to fsspec url_to_fs.
        """
        super().__init__(path)
        self.max_batch_size = max(1, max_batch_size)
        self.max_batch_bytes = max(1, max_batch_bytes)
        self.merge_item_bytes = merge_item_bytes
        self.range_bytes = max(1, range_bytes)
        if cpu_workers is None:
            local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", 1)))
            total_cpus = os.cpu_count() or 4
            cpu_workers = min(4, max(1, total_cpus // local_world_size))
        self.cpu_workers = max(1, cpu_workers)
        self.share_reads = share_reads
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    def _item_key(self, req: ReadItem) -> _ItemKey:
        md = self.storage_data[req.storage_index]
        return md.relative_path, md.offset, md.length

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        plan = super().prepare_local_plan(plan)
        if self.share_reads and plan.storage_data is None:
            plan = dataclasses.replace(plan, storage_data=_shm_host_id())
        return plan

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        plans = super().prepare_global_plan(plans)
        if not self.share_reads or not self._supports_batched_cat_ranges():
            return plans
        hosts: dict[str, list[int]] = {}
        free: dict[str, int] = {}
        for r, p in enumerate(plans):
            # read_data returns early on an empty plan, so it takes no part.
            if isinstance(p.storage_data, tuple) and p.items:
                host, host_free = p.storage_data
                hosts.setdefault(host, []).append(r)
                free[host] = min(free.get(host, host_free), host_free)
        prefix = os.path.join(_SHM_DIR, f"torch_dcp_{secrets.token_hex(16)}_")
        shares: list[_SharedReads | None] = [None] * len(plans)
        next_id = 0
        for host, ranks in hosts.items():
            if len(ranks) < 2:
                continue
            readers: dict[_ItemKey, set[int]] = {}
            for r in ranks:
                for req in plans[r].items:
                    key = self._item_key(req)
                    if key[2] > self.merge_item_bytes:
                        readers.setdefault(key, set()).add(r)
            shared = {k: sorted(v) for k, v in readers.items() if len(v) > 1}
            shared_bytes = sum(k[2] for k in shared)
            # Every shared item sits in /dev/shm until its last reader is done.
            if 2 * shared_bytes > free[host]:
                shared, shared_bytes = {}, 0
            send: dict[int, dict] = {r: {} for r in ranks}
            recv: dict[int, dict] = {r: {} for r in ranks}
            load = dict.fromkeys(ranks, 0)
            for r in ranks:
                for req in plans[r].items:
                    key = self._item_key(req)
                    if key not in shared:
                        load[r] += key[2]
            for key in sorted(shared, key=lambda k: -k[2]):
                owner = min(shared[key], key=lambda r: (load[r], r))
                load[owner] += key[2]
                others = [r for r in shared[key] if r != owner]
                send[owner][key] = (next_id, others)
                for r in others:
                    recv[r][key] = next_id
                next_id += 1
            tables = {}
            for r in ranks:
                batches, _ = self._batches(plans[r].items, send[r], recv[r])
                tables[r] = self._batch_table(batches)
            # So does a batch fetched for another rank, until that rank maps it.
            biggest = max(
                (sum(n for *_, n in b) for t in tables.values() for b in t), default=0
            )
            if 2 * (shared_bytes + len(ranks) * biggest) > free[host]:
                tables = dict.fromkeys(ranks, [])
            for r in ranks:
                shares[r] = _SharedReads(prefix, r, send[r], recv[r], tables, next_id)
        return [dataclasses.replace(p, storage_data=s) for p, s in zip(plans, shares)]

    def _batches(
        self,
        items: list[ReadItem],
        send: dict[_ItemKey, tuple[int, list[int]]],
        recv_keys: dict[_ItemKey, int],
    ) -> tuple[list, dict[_ItemKey, list[ReadItem]]]:
        """Split items into the cat_ranges batches read_data fetches, in order.

        A batch is a list of groups, and a group is one range holding one or
        more contiguous items: ``(relative_path, [(start, end)], members)``,
        where each member maps to (range index, lo, hi) segments of the fetched
        ranges. Items in ``recv_keys`` come from another rank and are returned
        apart, by key.
        """
        recv: dict[_ItemKey, list[ReadItem]] = {}
        keyed = []
        for req in items:
            key = self._item_key(req)
            if key in recv_keys:
                recv.setdefault(key, []).append(req)
            else:
                keyed.append((key not in send, key[0], key[1], req))
        # Items other ranks wait for are fetched first.
        reqs = [k[-1] for k in sorted(keyed, key=lambda k: k[:3])]

        groups = []
        span = None
        for req in reqs:
            item_md = self.storage_data[req.storage_index]
            path = item_md.relative_path
            start, end = item_md.offset, item_md.offset + item_md.length
            if item_md.length > self.merge_item_bytes:
                span = None
                groups.append((path, [(start, end)], [(req, [(0, 0, end - start)])]))
            elif (
                span is not None
                and span[0] == path
                and span[1][0][1] == start
                and end - span[1][0][0] <= self.range_bytes
            ):
                s0 = span[1][0][0]
                span[1][0] = (s0, end)
                span[2].append((req, [(0, start - s0, end - s0)]))
            else:
                span = (path, [(start, end)], [(req, [(0, 0, end - start)])])
                groups.append(span)

        batches = []
        batch = []
        n_ranges = 0
        batch_bytes = 0
        for g in groups:
            g_bytes = sum(e - s for s, e in g[1])
            if batch and (
                n_ranges + len(g[1]) > self.max_batch_size
                or batch_bytes + g_bytes > self.max_batch_bytes
            ):
                batches.append(batch)
                batch = []
                n_ranges = 0
                batch_bytes = 0
            batch.append(g)
            n_ranges += len(g[1])
            batch_bytes += g_bytes
        if batch:
            batches.append(batch)
        return batches, recv

    @staticmethod
    def _batch_table(batches: list) -> list[list[_ItemKey]]:
        return [[(p, s, e - s) for p, rs, _ in b for s, e in rs] for b in batches]

    def _supports_batched_cat_ranges(self) -> bool:
        if not (self.fs and self.fs.fs and hasattr(self.fs.fs, "cat_ranges")):
            return False
        # AsyncFileSystem subclasses (gcsfs, s3fs) bind the sync cat_ranges onto
        # the instance via mirror_sync_methods, so it is not on the class.
        if isinstance(self.fs.fs, fsspec.asyn.AsyncFileSystem):
            return True
        # The AbstractFileSystem fallback reopens the file per range, which is
        # slower than the single stream per shard in FileSystemReader.read_data.
        cat_ranges_fn = getattr(
            self.fs.fs.cat_ranges, "__func__", self.fs.fs.cat_ranges
        )
        return cat_ranges_fn is not fsspec.AbstractFileSystem.cat_ranges

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        share = plan.storage_data
        if not isinstance(share, _SharedReads):
            share = _SharedReads("", -1, {}, {}, {}, 0)
        host = share.batches
        batches = None
        recv: dict[_ItemKey, list[ReadItem]] = {}
        if plan.items and self._supports_batched_cat_ranges():
            batches, recv = self._batches(plan.items, share.send, share.recv)
        mine = host.get(share.rank, [])
        claims = batches is not None and mine == self._batch_table(batches)
        if host and not claims:
            # Keep the other ranks from taking batches this rank will not map,
            # e.g. because finish_plan changed its reads after the coordinator
            # split them up.
            for i in range(len(mine)):
                with suppress(OSError):
                    _create_excl(f"{share.prefix}c{share.rank}.{i}")
        if batches is None:
            if host:
                _leave_host(share)
            return super().read_data(plan, planner)
        victims = {r: t for r, t in host.items() if r != share.rank and t}

        recv_batches = []
        batch = []
        batch_bytes = 0
        for key, key_reqs in recv.items():
            if batch and (
                len(batch) >= self.max_batch_size
                or batch_bytes + key[2] > self.max_batch_bytes
            ):
                recv_batches.append(batch)
                batch = []
                batch_bytes = 0
            batch.append((key, key_reqs))
            batch_bytes += key[2]
        if batch:
            recv_batches.append(batch)

        published: set[_ItemKey] = set()
        publishes: list[concurrent.futures.Future] = []
        deadline = time.monotonic() + _SHARE_TIMEOUT_S
        own_done = threading.Event()
        stop = threading.Event()

        def fetch_ranges(ranges):
            paths = [self.fs.concat_path(self.path, rel) for rel, _, _ in ranges]
            starts = [s for _, s, _ in ranges]
            ends = [e for _, _, e in ranges]
            chunks = self.fs.fs.cat_ranges(paths, starts, ends, on_error="raise")
            # A short list means some ranges were dropped (``on_error="omit"``).
            # Left unchecked, items would silently be skipped and leave their
            # tensors at whatever the caller initialized them to.
            if len(chunks) != len(paths):
                raise RuntimeError(
                    f"cat_ranges returned {len(chunks)} chunks for {len(paths)} ranges"
                )
            # ``on_error`` is advisory: fsspec honors it only since 2026.7.0 and
            # other backends may ignore it, returning exceptions in-band.
            for path, start, end, chunk in zip(paths, starts, ends, chunks):
                if isinstance(chunk, BaseException):
                    raise RuntimeError(
                        f"Failed to read bytes [{start}, {end}) from {path}"
                    ) from chunk
                if len(chunk) != end - start:
                    raise RuntimeError(
                        f"Read {len(chunk)} bytes for [{start}, {end}) from {path}"
                    )
            return chunks

        def batch_items(b, chunks):
            items = []
            k = 0
            for _, ranges, members in b:
                views = [memoryview(c) for c in chunks[k : k + len(ranges)]]
                k += len(ranges)
                for req, segs in members:
                    items.append((req, [views[i][lo:hi] for i, lo, hi in segs]))
                key = self._item_key(members[0][0])
                if key in share.send and key not in published:
                    published.add(key)
                    name, readers = share.send[key]
                    publishes.append(
                        publish_executor.submit(
                            _publish,
                            f"{share.prefix}{name}.tmp{share.rank}",
                            [f"{share.prefix}{name}.{r}" for r in readers],
                            [chunks[k - 1]],
                        )
                    )
            return items

        def fetch_batch(b):
            ranges = [(rel, s, e) for rel, rs, _ in b for s, e in rs]
            return batch_items(b, fetch_ranges(ranges))

        def fetch_own(i):
            try:
                # A rank that cannot create its claim file reads the batch anyway.
                with suppress(OSError):
                    if claims and not _create_excl(f"{share.prefix}c{share.rank}.{i}"):
                        return i
                return fetch_batch(batches[i])
            finally:
                if i == len(batches) - 1:
                    own_done.set()

        def receive_helped(i):
            b = batches[i]
            lengths = [e - s for _, rs, _ in b for s, e in rs]
            name = f"{share.prefix}h{share.rank}.{i}"
            data = _receive_shared(name, sum(lengths), deadline, stop)
            if data is None:
                return [] if stop.is_set() else fetch_batch(b)
            view = memoryview(data)
            offsets = itertools.accumulate(lengths, initial=0)
            return batch_items(b, [view[o : o + n] for o, n in zip(offsets, lengths)])

        def receive_batch(b):
            items = []
            missing = []
            for key, key_reqs in b:
                path = f"{share.prefix}{share.recv[key]}.{share.rank}"
                data = _receive_shared(path, key[2], deadline, stop)
                if data is None:
                    missing.append((key, key_reqs))
                    continue
                view = memoryview(data)
                items.extend((req, [view]) for req in key_reqs)
            if missing and not stop.is_set():
                items += fetch_batch(
                    [
                        (
                            key[0],
                            [(key[1], key[1] + key[2])],
                            [(req, [(0, 0, key[2])]) for req in key_reqs],
                        )
                        for key, key_reqs in missing
                    ]
                )
            return items

        def steal():
            own_done.wait()
            last = {r: len(t) - 1 for r, t in victims.items()}
            sizes = {r: [sum(n for *_, n in b) for b in t] for r, t in victims.items()}

            def claimed(r, i):
                return os.path.exists(f"{share.prefix}c{r}.{i}")

            def handover_late(r, i, left):
                # Batch i is in flight and the one after it is the last left.
                # Ranks claim a batch as the previous one lands, so the claim
                # times give the rank's speed and how long i has to go.
                try:
                    t0 = os.stat(f"{share.prefix}c{r}.{i - 1}").st_mtime
                    t1 = os.stat(f"{share.prefix}c{r}.{i}").st_mtime
                except OSError:
                    return False
                eta = (t1 - t0) * sizes[r][i] / max(1, sizes[r][i - 1])
                togo = eta - (time.time() - t1)
                return togo < left / _HANDOVER_BYTES_PER_S + _HANDOVER_S

            while not stop.is_set():
                # Ranks claim their own batches from the front and helpers take
                # them from the back, so the unclaimed ones are contiguous.
                best, most = None, 0
                for r in victims:
                    i = last[r]
                    while i >= 0 and claimed(r, i):
                        i -= 1
                    last[r] = i
                    left = 0
                    while i >= 0 and not claimed(r, i):
                        left += sizes[r][i]
                        i -= 1
                    if last[r] - i == 1 and i >= 1 and handover_late(r, i, left):
                        continue
                    if left > most:
                        best, most = r, left
                if best is None:
                    return
                i = last[best]
                try:
                    if not _create_excl(f"{share.prefix}c{best}.{i}"):
                        continue
                except OSError:
                    return
                name = f"{share.prefix}h{best}.{i}"
                try:
                    chunks = fetch_ranges(
                        [(rel, s, s + n) for rel, s, n in victims[best][i]]
                    )
                except Exception as e:
                    # An empty file sends that rank back to storage.
                    logger.warning("Could not read a batch for rank %d: %s", best, e)
                    chunks = []
                publishes.append(
                    publish_executor.submit(_publish, f"{name}.tmp", [name], chunks)
                )

        layouts: dict[tuple[bytes, int], tuple] = {}

        def decode(req, segs):
            item_md = self.storage_data[req.storage_index]
            if (
                req.type == LoadItemType.BYTE_IO
                or item_md.transform_descriptors
                or len(segs) != 1
            ):
                return self._decode_item(req, _SegmentReader(segs))
            # Storages alias the fetched bytes rather than being copied out of
            # them under the GIL, so ``dst.copy_`` reads the network buffer.
            tensor = _load_aliased(segs[0], layouts)
            return narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)

        def process(items):
            b_reqs = [req for req, _ in items]
            decoded = [cpu_executor.submit(decode, req, segs) for req, segs in items]
            # The futures below own their buffers now; holding the list too
            # would pin every raw buffer for the whole batch.
            items.clear()

            # Every planner hook runs on this thread, so planners need not be
            # thread safe. Only torch.load above and the copies below go to
            # the pool.
            pending: list[tuple[ReadItem, Tensor, Tensor]] = []
            for i, req in enumerate(b_reqs):
                f = decoded[i]
                # Drop the future so a completed one stops pinning its decoded
                # tensor for the rest of the batch.
                decoded[i] = None
                item = f.result()
                if req.type == LoadItemType.BYTE_IO:
                    planner.load_bytes(req, item)
                else:
                    pending.append((req, self._resolve_item(req, item, planner), item))

            if len(pending) > 1 and _destinations_disjoint(
                [dst for _, dst, _ in pending]
            ):
                copies = [cpu_executor.submit(dst.copy_, src) for _, dst, src in pending]
                for c in copies:
                    c.result()
                for req, dst, _ in pending:
                    planner.commit_tensor(req, dst)
            else:
                # Overlapping destinations can mean the planner handed back one
                # staging buffer, so each item has to be copied and committed
                # before the next is touched.
                for req, dst, src in pending:
                    cpu_executor.submit(dst.copy_, src).result()
                    planner.commit_tensor(req, dst)

        # Opening a remote file costs a metadata lookup and a stream setup
        # before its first byte arrives. Opening the files this rank may read
        # after its first batch, its own and those of the ranks it may help,
        # while that batch downloads keeps these round trips out of the later
        # batches.
        warm = []
        if batches and isinstance(self.fs.fs, fsspec.asyn.AsyncFileSystem):
            first = {rel for rel, _, _ in batches[0]}
            later = itertools.chain(
                (rel for b in batches[1:] for rel, _, _ in b),
                (rel for t in victims.values() for b in t for rel, _, _ in b),
            )
            warm = [rel for rel in dict.fromkeys(later) if rel not in first]
            warm = warm[:_WARM_FILES]

        def warm_files():
            paths = [self.fs.concat_path(self.path, rel) for rel in warm]
            n = len(paths)
            # Best effort: the batches read these files either way.
            with suppress(Exception):
                self.fs.fs.cat_ranges(paths, [0] * n, [1] * n, on_error="return")

        jobs = collections.deque((fetch_own, i) for i in range(len(batches)))
        if not batches:
            own_done.set()
        intra_op_threads = torch.get_num_threads()
        try:
            with (
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.cpu_workers,
                    # Copies already run in parallel on these threads. Giving
                    # each one a full intra-op pool as well oversubscribes a host
                    # shared by several ranks and slows their downloads.
                    initializer=torch.set_num_threads,
                    initargs=(1,),
                ) as cpu_executor,
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch_executor,
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as recv_executor,
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as publish_executor,
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as steal_executor,
            ):
                helping = steal_executor.submit(steal) if victims else None
                # Items from other ranks are mapped as they arrive and copied
                # while this rank's own batches download.
                received = collections.deque(
                    recv_executor.submit(receive_batch, b) for b in recv_batches
                )
                inflight: collections.deque[concurrent.futures.Future] = (
                    collections.deque()
                )

                def top_up():
                    # With one fetch queued behind the running one, the next
                    # starts as soon as it ends, even while this thread is still
                    # busy with a large copy.
                    while jobs and len(inflight) < 2:
                        inflight.append(prefetch_executor.submit(*jobs.popleft()))

                try:
                    top_up()
                    if warm:
                        threading.Thread(target=warm_files, daemon=True).start()
                    while inflight or received:
                        waiting = [received[0]] if received else []
                        if inflight:
                            waiting.append(inflight[0])
                        concurrent.futures.wait(
                            waiting, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        if not inflight or not inflight[0].done():
                            process(received.popleft().result())
                            continue
                        items = inflight.popleft().result()
                        if isinstance(items, int):
                            # Another rank took this batch and hands it over.
                            jobs.append((receive_helped, items))
                            items = None
                        top_up()
                        if items is not None:
                            process(items)

                    if helping is not None:
                        helping.result()
                    # Readers fall back to storage for anything not handed over.
                    for f in publishes:
                        if (e := f.exception()) is not None:
                            logger.warning(
                                "Could not share a read via %s: %s", _SHM_DIR, e
                            )
                except BaseException:
                    stop.set()
                    own_done.set()
                    raise
                finally:
                    # __exit__ calls shutdown(wait=True) without ``cancel_futures``,
                    # so on failure it would drain the queue instead of dropping
                    # it. Waiting is left to __exit__; queued prefetches are
                    # cancelled so their results are not silently discarded.
                    for f in inflight:
                        f.cancel()
                    cpu_executor.shutdown(wait=False, cancel_futures=True)
                    recv_executor.shutdown(wait=False, cancel_futures=True)
        finally:
            if host:
                _leave_host(share)
            # The pool's set_num_threads also changed the default for threads
            # that have not run an op yet.
            torch.set_num_threads(intra_op_threads)

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)
