# Synchronous Training and Throughput Metrics Design

## Goal

Restore the known-good synchronous batch construction behavior from version
1.6.1 and expose unambiguous data, compute, and end-to-end throughput metrics.

## Scope

Only the training defaults and throughput reporting change. The model,
auxiliary tasks, cache schema, dataset splits, validation, optimizer, and
checkpoint format remain unchanged.

## Batch Preparation

The default configuration is:

```yaml
train:
  prefetch_workers: 0
  prefetch_batches: 0
```

This makes the main process collate one batch synchronously and then train that
batch before collating the next. Existing thread and process modes remain
available for controlled experiments but are not enabled by default.

## Metrics

All counters accumulate across the complete training run, including epoch
boundaries:

- `train/data_samples_per_second`: processed samples divided by cumulative time
  spent obtaining the next batch.
- `train/compute_samples_per_second`: processed samples divided by cumulative
  `train_batch` time.
- `train/end_to_end_samples_per_second`: processed samples divided by cumulative
  data plus compute time.
- `train/samples_per_second`: compatibility alias for end-to-end throughput.
- `train/data_wait_ratio`: cumulative data time divided by cumulative data plus
  compute time.

The console log prints `data_samples/s`, `compute_samples/s`, and `samples/s`,
where `samples/s` is end-to-end throughput. If cumulative data time is zero, the
data-only throughput is reported as zero rather than infinity.

## Testing

Unit tests verify the three throughput formulas, the zero-data-time case, and
the compatibility alias. Configuration parsing verifies that the checked-in
defaults select synchronous collation. Existing batch-prefetch tests remain to
ensure optional modes still work.
