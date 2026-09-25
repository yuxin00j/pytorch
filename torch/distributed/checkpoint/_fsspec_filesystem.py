# Mypy will not try inferring the types of any 3rd party libraries installed.
# mypy: ignore-errors

import bisect
import concurrent.futures
import io
import itertools
import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec
import fsspec.asyn
from fsspec.core import url_to_fs

from torch import Tensor
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


if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


__all__ = [
    "FsspecWriter",
    "FsspecReader",
]


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


class FsspecReader(FileSystemReader):
    def __init__(
        self,
        path: str | os.PathLike,
        *,
        max_batch_size: int = 64,
        max_batch_bytes: int = 256 * 1024 * 1024,
        merge_item_bytes: int = 1024 * 1024,
        range_bytes: int = 16 * 1024 * 1024,
        cpu_workers: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the FsspecReader pointing to `path`.

        Args:
            path: directory or URL where the checkpoint will be read from.
            max_batch_size: Maximum number of ranges per batched cat_ranges call.
                Defaults to 64.
            max_batch_bytes: Maximum cumulative byte size requested per batched
                cat_ranges call. Defaults to 256 MiB. This caps one request, not
                resident memory: the next batch is fetched while the current one is
                still being decoded and copied, so expect a small multiple of this
                to be live at peak.
            merge_item_bytes: Contiguous items no larger than this are merged
                into one range, since each range has a fixed cost that dominates
                for small items. Larger items are read as their own range.
                Defaults to 1 MiB.
            range_bytes: Maximum size of a merged range. Defaults to 16 MiB.
            cpu_workers: Number of worker threads for parallel CPU deserialization.
                Defaults to min(4, max(1, cpu_count // local_world_size)).
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
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

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
        if not plan.items or not self._supports_batched_cat_ranges():
            return super().read_data(plan, planner)

        reqs = sorted(
            plan.items,
            key=lambda req: (
                self.storage_data[req.storage_index].relative_path,
                self.storage_data[req.storage_index].offset,
            ),
        )

        # A group is one range holding one or more contiguous items. Each
        # member maps to (range index, lo, hi) segments of the fetched ranges.
        groups = []
        span = None
        for req in reqs:
            item_md = self.storage_data[req.storage_index]
            path = self.fs.concat_path(self.path, item_md.relative_path)
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

        def fetch_batch(b):
            bp = [path for path, ranges, _ in b for _ in ranges]
            bs = [s for _, ranges, _ in b for s, _ in ranges]
            be = [e for _, ranges, _ in b for _, e in ranges]
            chunks = self.fs.fs.cat_ranges(bp, bs, be, on_error="raise")
            # A short list means some ranges were dropped (``on_error="omit"``).
            # Left unchecked, items would silently be skipped and leave their
            # tensors at whatever the caller initialized them to.
            if len(chunks) != len(bp):
                raise RuntimeError(
                    f"cat_ranges returned {len(chunks)} chunks for {len(bp)} ranges"
                )
            # ``on_error`` is advisory: fsspec honors it only since 2026.7.0 and
            # other backends may ignore it, returning exceptions in-band.
            for path, start, end, chunk in zip(bp, bs, be, chunks):
                if isinstance(chunk, BaseException):
                    raise RuntimeError(
                        f"Failed to read bytes [{start}, {end}) from {path}"
                    ) from chunk
                if len(chunk) != end - start:
                    raise RuntimeError(
                        f"Read {len(chunk)} bytes for [{start}, {end}) from {path}"
                    )
            items = []
            k = 0
            for _, ranges, members in b:
                views = [memoryview(c) for c in chunks[k : k + len(ranges)]]
                k += len(ranges)
                for req, segs in members:
                    items.append((req, [views[i][lo:hi] for i, lo, hi in segs]))
            return items

        def decode(req, segs):
            return self._decode_item(req, _SegmentReader(segs))

        with (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=self.cpu_workers
            ) as cpu_executor,
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch_executor,
        ):
            next_io: concurrent.futures.Future | None = None
            try:
                next_io = prefetch_executor.submit(fetch_batch, batches[0])

                for idx in range(len(batches)):
                    items = next_io.result()
                    next_io = None

                    if idx + 1 < len(batches):
                        next_io = prefetch_executor.submit(
                            fetch_batch, batches[idx + 1]
                        )

                    b_reqs = [req for req, _ in items]
                    decoded = [
                        cpu_executor.submit(decode, req, segs) for req, segs in items
                    ]
                    # The futures below own their buffers now; holding the list
                    # too would pin every raw buffer for the whole batch.
                    del items

                    # Every planner hook runs on this thread, so planners need
                    # not be thread safe. Only torch.load above and the copies
                    # below go to the pool.
                    pending: list[tuple[ReadItem, Tensor, Tensor]] = []
                    for i, req in enumerate(b_reqs):
                        f = decoded[i]
                        # Drop the future so a completed one stops pinning its
                        # decoded tensor for the rest of the batch.
                        decoded[i] = None
                        item = f.result()
                        if req.type == LoadItemType.BYTE_IO:
                            planner.load_bytes(req, item)
                        else:
                            pending.append(
                                (req, self._resolve_item(req, item, planner), item)
                            )

                    if len(pending) > 1 and _destinations_disjoint(
                        [dst for _, dst, _ in pending]
                    ):
                        copies = [
                            cpu_executor.submit(dst.copy_, src)
                            for _, dst, src in pending
                        ]
                        for c in copies:
                            c.result()
                        for req, dst, _ in pending:
                            planner.commit_tensor(req, dst)
                    else:
                        # Overlapping destinations can mean the planner handed
                        # back one staging buffer, so each item has to be
                        # copied and committed before the next is touched.
                        for req, dst, src in pending:
                            dst.copy_(src)
                            planner.commit_tensor(req, dst)
            finally:
                # __exit__ calls shutdown(wait=True) without ``cancel_futures``,
                # so on failure it would drain the queue instead of dropping it.
                # Waiting is left to __exit__; the in-flight prefetch is
                # cancelled so its result is not silently discarded.
                if next_io is not None:
                    next_io.cancel()
                cpu_executor.shutdown(wait=False, cancel_futures=True)

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)
