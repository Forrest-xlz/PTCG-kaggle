# Multi-process mmap Batch Prefetch Design

## Goal

Reduce GPU idle time caused by the single-threaded Python `collate` loop. The
training sample order, shuffle result, model inputs, losses, validation splits,
and cache format must remain unchanged.

## Configuration

Add `train.prefetch_workers` and retain `train.prefetch_batches`:

- `prefetch_workers: 0`: synchronous batching without prefetch.
- `prefetch_workers: 1`: current bounded background-thread prefetch.
- `prefetch_workers >= 2`: ordered multi-process batching.
- `prefetch_batches`: maximum number of submitted or completed batches waiting
  outside the GPU training loop. It bounds extra host-memory usage.

The initial recommended configuration is four workers and eight prefetched
batches.

## Architecture

The main process generates `IndexBatch` objects using the existing
`iter_index_batches` method. This preserves the exact seeded shuffle and batch
boundaries.

For multi-process mode, workers are initialized with the cache root and expected
feature signature. Each worker opens its own read-only `MmapFeatureDataset` once
and reuses it for all assigned batches. A worker receives only a batch sequence
number and its array of global sample IDs, calls the existing `collate`, and
returns the sequence number plus `CachedBatch`.

The main process keeps at most `prefetch_batches` futures in flight. Results are
consumed strictly in submission order, even when later batches finish first.
Therefore multiprocessing cannot change training order. GPU operations remain
exclusively in the main process; workers never initialize CUDA.

The multiprocessing context uses `spawn`, avoiding unsafe CUDA state inherited
through `fork`. The process pool is created for an epoch and closed at the end of
that epoch. Worker mmap handles are released when workers exit.

## Failure Handling

Worker exceptions propagate through the corresponding future. The main process
cancels outstanding futures and shuts down the executor before re-raising the
original failure. Empty datasets terminate normally. Keyboard interruption also
closes the pool.

## Measurement

The existing `data_wait_ratio` remains the primary measurement. It includes time
spent waiting for the next ordered prefetched result. Overall `samples/s` also
continues to include both data wait and GPU compute time.

The first batch includes process startup and mmap initialization, so performance
should be judged after at least 50-100 optimizer steps. If data wait is low while
GPU utilization remains intermittent, the next investigation should measure H2D,
forward, backward, and optimizer phases separately.

## Resource Trade-offs

Each worker owns Python dataset metadata and mmap objects, while file-backed pages
remain shareable through the operating-system page cache. Returned NumPy batches
must cross process boundaries, so multiprocessing adds serialization/shared-memory
traffic. `prefetch_batches` bounds the number of resident completed batches.

Four workers are the initial default. More workers are not assumed to be faster;
disk bandwidth and inter-process transfer may become the bottleneck.

## Testing

Tests cover:

- ordered output when workers complete out of order;
- preservation of index-batch order;
- worker exception propagation;
- zero-worker and one-worker fallback behavior;
- configuration validation;
- existing training, cache, and auxiliary-task regressions.

No extraction or cache rebuild is required.
