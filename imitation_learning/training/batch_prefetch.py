"""Bounded CPU collation for cached training batches."""
from __future__ import annotations

import multiprocessing
import queue
import threading
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

from training.feature_cache import (
    CachedBatch,
    IndexBatch,
    MmapFeatureDataset,
)


_WORKER_DATASET: MmapFeatureDataset | None = None


def validate_prefetch_settings(workers: int, buffer_size: int) -> None:
    if type(workers) is not int or workers < 0:
        raise ValueError("prefetch_workers must be an integer >= 0")
    if type(buffer_size) is not int or buffer_size < 0:
        raise ValueError("prefetch_batches must be an integer >= 0")


def prefetch_iterable(iterable: Iterable, buffer_size: int) -> Iterator:
    """Prepare upcoming items on one bounded background thread."""
    if buffer_size < 1:
        yield from iterable
        return
    buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
    sentinel = object()

    def produce() -> None:
        try:
            for item in iterable:
                buffer.put((item, None))
        except BaseException as exc:
            buffer.put((sentinel, exc))
        else:
            buffer.put((sentinel, None))

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    while True:
        item, error = buffer.get()
        if item is sentinel:
            if error is not None:
                raise error
            return
        yield item


def _initialize_worker(cache_root: str, expected_signature: dict) -> None:
    global _WORKER_DATASET
    _WORKER_DATASET = MmapFeatureDataset(
        Path(cache_root), expected_signature=expected_signature
    )


def _collate_in_worker(
    sequence: int,
    global_ids,
) -> tuple[int, CachedBatch]:
    if _WORKER_DATASET is None:
        raise RuntimeError("batch worker dataset was not initialized")
    return sequence, _WORKER_DATASET.collate(IndexBatch(global_ids))


def _iter_process_batches(
    cache_root: Path,
    expected_signature: dict,
    index_batches: Iterable[IndexBatch],
    workers: int,
    buffer_size: int,
) -> Iterator[CachedBatch]:
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(str(cache_root), expected_signature),
    )
    pending: deque[tuple[int, Future]] = deque()
    source = iter(enumerate(index_batches))

    def submit_one() -> bool:
        try:
            sequence, index_batch = next(source)
        except StopIteration:
            return False
        pending.append(
            (
                sequence,
                executor.submit(
                    _collate_in_worker,
                    sequence,
                    index_batch.global_ids,
                ),
            )
        )
        return True

    try:
        for _ in range(buffer_size):
            if not submit_one():
                break
        while pending:
            expected_sequence, future = pending.popleft()
            actual_sequence, batch = future.result()
            if actual_sequence != expected_sequence:
                raise RuntimeError(
                    "prefetched batch sequence mismatch: "
                    f"expected {expected_sequence}, got {actual_sequence}"
                )
            submit_one()
            yield batch
    finally:
        for _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def iter_collated_batches(
    dataset: MmapFeatureDataset,
    index_batches: Iterable[IndexBatch],
    *,
    workers: int,
    buffer_size: int,
    expected_signature: dict,
) -> Iterator[CachedBatch]:
    """Collate batches synchronously, in a thread, or in ordered processes."""
    validate_prefetch_settings(workers, buffer_size)
    if workers == 0 or buffer_size == 0:
        for index_batch in index_batches:
            yield dataset.collate(index_batch)
        return
    if workers == 1:
        yield from prefetch_iterable(
            (dataset.collate(index_batch) for index_batch in index_batches),
            buffer_size,
        )
        return
    yield from _iter_process_batches(
        dataset.root,
        expected_signature,
        index_batches,
        workers,
        buffer_size,
    )
