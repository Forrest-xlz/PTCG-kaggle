# Mmap Feature Cache and Mixed-Precision Training Design

## Goal

Scale pure behavior-cloning training to tens of millions of winner-only replay
samples without retaining one Python object graph per sample. Preserve the
existing encoder/decoder feature semantics and policy loss while allowing the
training compute precision to be selected from FP32, FP16, and BF16.

## Scope

This change adds a separate, resumable feature-cache build step between replay
extraction and training:

```text
replay ZIP
  -> training.extract
  -> winner-only JSONL shards
  -> training.cache_features
  -> flat mmap feature shards
  -> training.train
```

The replay extractor remains responsible only for selecting winner decisions
and writing replay records. Feature construction remains in
`model/features.py`. The cache builder calls those existing feature functions;
it does not define a second feature implementation.

The Kaggle submission notebook and inference-time feature construction are not
changed.

## Alternatives Considered

### A. Flat, sharded mmap cache (selected)

Build model-ready flat binary arrays once, memory-map them during training, and
keep only the current batch plus a bounded per-shard permutation in process
memory. This adds one preprocessing command but removes per-sample Python
objects and repeated JSON/gzip/deserialization work from every training run.

### B. Flat arrays constructed at training startup

This avoids an explicit cache command but repeats expensive feature generation
on every run and still requires large resident arrays. It does not meet the
tens-of-millions scale target.

### C. Change only float32 values to float16

This halves only value storage while retaining six arrays and three dataclass
objects per sample. At tens of millions of samples, object overhead alone is
several gigabytes, so this is insufficient.

## Configuration

Add `cfg/cache.yaml`:

```yaml
cache:
  cg_path: ../pokemon_tcg_ai_battle/sample_submission
  input: data/training
  output: data/training-cache
  workers: 4
  samples_per_shard: 250000
  force: false
```

The cache builder has no command-line configuration and is run from the
`imitation_learning` directory:

```bash
python -m training.cache_features
```

Update `cfg/train.yaml`:

```yaml
train:
  data: data/training-cache
  precision: bf16  # fp32, fp16, or bf16
  shuffle_mode: global  # global or shard
```

The existing JSONL preload settings are removed because mmap training neither
preloads all samples nor streams raw JSON:

- `preload`
- `preload_workers`
- `preload_chunk_size`
- `shuffle_buffer`

`max_samples` remains supported and limits samples consumed per epoch.

## Cache Shard Layout

Each source `<date>.jsonl.gz` produces one or more atomic cache directories,
with no directory exceeding `cache.samples_per_shard` representable samples.
A cache directory contains one `data.bin` with aligned array sections and one
`meta.json` describing each section's byte offset, dtype, shape, and byte
length. Packing the arrays into one binary file keeps the number of open mmap
file descriptors proportional to the number of active shards rather than the
number of arrays. The logical sections are:

| Section | Type | Meaning |
|---|---|---|
| `encoder_index` | uint16 | Flat encoder feature indices |
| `encoder_value` | float16 | Flat encoder per-sample weights |
| `encoder_ptr` | uint32 | Per-sample boundaries into encoder arrays |
| `encoder_offset` | uint16 | 24 local EmbeddingBag word offsets per sample |
| `decoder_index` | uint32 | Flat decoder feature indices |
| `decoder_ptr` | uint32 | Per-sample boundaries into decoder indices |
| `decoder_offset` | uint16 | Flat local decoder word offsets |
| `decoder_offset_ptr` | uint32 | Per-sample boundaries into decoder offsets |
| `target` | uint8 | Selected candidate index |
| `action_count` | uint8 | Number of valid candidate actions |

The decoder value array is omitted. The current decoder feature builder emits
only weight `1`, so `EmbeddingBag` receives `per_sample_weights=None`.

Pointers are shard-local uint32 values. The builder rejects a shard if any flat
array would exceed the uint32 range. Sharding already follows replay archives,
so normal shards are far below this limit.

Encoder indices are uint16 because the encoder vocabulary is fixed at 22,000.
Decoder indices remain uint32 because the decoder vocabulary exceeds 65,535.
Local offsets are uint16 and are range-checked before writing. Targets and
action counts are uint8 because policy candidates are capped at 64.

Encoder weights use float16 storage. They are converted to the configured
compute dtype while assembling a batch. Cache metadata records the storage
dtype so later schema changes cannot be silently misread.

The cache builder uses `cache.cg_path` to load the competition API and derive
card and attack counts. NumPy is an explicit project dependency because raw
array creation, mmap access, bounded permutations, and batch concatenation all
use NumPy dtypes and buffers.

## Atomicity and Resumability

Workers process different source JSONL shards independently. Each worker splits
its output at `cache.samples_per_shard`, writes each part to a temporary sibling
directory, flushes and closes all files, writes `meta.json` last, and atomically
renames the directory to its final name.

A completed shard is skipped only when all of these match:

- cache schema version;
- source path identity, size, and modification time;
- source extraction metadata reports `winner_only: true`;
- card count, attack count, encoder size, decoder size, and maximum actions;
- all declared files exist with their expected byte sizes.

Invalid, partial, stale, or incompatible shards are rebuilt. `force: true`
always rebuilds them.

## Training Data Flow and Shuffle

Training discovers and validates all cache shards before creating the model.
It memory-maps arrays read-only. It does not create a `PreparedSample` for every
record.

For `shuffle_mode: global`, each epoch creates a compact uint32 global sample
permutation when the total sample count is below 2^32, maps each global ID to a
physical shard and local ID, and assembles batches from those locations. For
30 million samples the permutation occupies about 120 MB. Larger datasets use
uint64 IDs. This gives a true global shuffle while retaining physical shards
for resumability and bounded cache construction.

For `shuffle_mode: shard`, each epoch shuffles shard order and creates a uint32
permutation only for the current shard. Every shard is consumed exactly once
per global epoch. Optimizer state, scheduler state, and global step continue
across shard boundaries.

Both modes concatenate selected mmap slices into bounded batch arrays, convert
indices and offsets to matching int32 tensors, convert encoder weights to the
selected floating compute dtype, and transfer only the batch to the configured
device. Global shuffle is the default; shard shuffle is the fallback when
random mmap access limits storage throughput.

The scheduler's total step count uses the validated cache metadata, capped by
`max_samples` when configured.

## Mixed Precision

The model keeps FP32 master parameters and FP32 checkpoints.

### FP32

- No autocast.
- No gradient scaler.
- Existing numerical behavior is preserved.

### FP16

- CUDA autocast uses `torch.float16`.
- `torch.amp.GradScaler` scales the loss.
- Gradients are unscaled before gradient clipping.
- If the scaler skips an optimizer step because of non-finite gradients, the
  learning-rate scheduler does not advance.

### BF16

- CUDA autocast uses `torch.bfloat16`.
- No gradient scaler is used.
- Startup fails with an actionable error when the selected CUDA device does
  not support BF16.

CPU training supports only `precision: fp32` in this project. Unsupported
device/precision combinations fail during configuration validation instead of
silently falling back.

W&B and console metrics continue to include loss, accuracy, gradient norm,
learning rate, throughput, epoch, and optimizer step. Mixed-precision runs also
log the configured precision, gradient scale for FP16, and skipped optimizer
steps.

## Correctness and Error Handling

Cache construction validates before narrowing every dtype:

- encoder index less than 65,536;
- local offsets less than 65,536;
- pointers less than 2^32;
- target less than action count and both at most 64;
- finite encoder weights representable as float16.

Errors identify the source shard and record number. A failing worker leaves no
completed cache directory.

Training validates metadata and physical byte sizes for every shard before the
first epoch. Cache schema or feature-constant mismatches instruct the user to
rerun `python -m training.cache_features`.

## Tests

Tests use a small synthetic cache and representative real replay records:

1. Compare legacy `prepare_sample` output with a cache round trip.
2. Verify all indices, offsets, targets, and action counts are identical.
3. Verify encoder float16 values are within the expected conversion tolerance.
4. Verify omitting decoder weights produces the same decoder EmbeddingBag
   output as explicit all-one weights.
5. Verify dtype overflow and stale metadata fail with clear errors.
6. Verify shard resume and force-rebuild behavior.
7. Verify `samples_per_shard` bounds every completed cache part.
8. Verify global shuffle visits every sample exactly once and shard shuffle
   visits every shard exactly once per global epoch.
9. Run one optimizer step in FP32.
10. Run CUDA FP16 and BF16 smoke tests when the current device supports them;
   otherwise assert the documented validation behavior.
11. Verify scheduler advancement is skipped together with an overflowed FP16
   optimizer step.

## Expected Resource Impact

Relative to the current per-sample compact-array preload:

- raw cached feature bytes are expected to fall roughly 40-60%;
- per-sample Python object overhead is eliminated;
- resident CPU memory becomes bounded by mmap working pages, the current
  shard permutation, and a small number of batches;
- repeated gzip, JSON, typed-observation, and feature-construction work moves
  from every training run into one resumable cache build.

Exact disk size, resident memory, cache-build throughput, and train throughput
will be reported during cache construction and training so the estimates can
be checked on the real dataset.
