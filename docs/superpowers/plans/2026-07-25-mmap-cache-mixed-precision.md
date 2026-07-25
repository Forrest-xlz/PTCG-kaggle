# Mmap Feature Cache and Mixed-Precision Training Implementation Plan

**Goal:** Replace all-sample Python preloading with sharded mmap feature caches,
support global or shard shuffle, and add configurable FP32/FP16/BF16 training.

**Architecture:** `training.cache_features` converts winner-only JSONL shards
into bounded flat binary cache parts. `training.feature_cache` owns the packed
format, mmap reader, global/shard permutations, and batch collation.
`training.train` consumes only cached batches and applies AMP according to YAML.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, PyYAML, pytest.

## Global Constraints

- Do not implement validation-set splitting in this change.
- Preserve encoder and decoder feature semantics and the pure policy loss.
- Decoder values are implicit ones and are not stored.
- Cache parts contain at most `cache.samples_per_shard` samples.
- Physical shards support both true global shuffle and shard-local shuffle.
- Checkpoints retain FP32 model state for Kaggle compatibility.

---

### Task 1: Packed feature-cache format

**Files:**
- Create: `imitation_learning/training/feature_cache.py`
- Create: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- `FeatureRecord`: encoder/decoder sparse arrays, target, action count.
- `PackedShardWriter.add(record)` and `finalize()`.
- `PackedShard(path)` exposes mmap-backed logical arrays.
- `CachedBatch` contains concatenated NumPy arrays for one training batch.

- [ ] Write tests that round-trip two synthetic sparse records and assert
  indices, offsets, FP16 values, targets, and action counts.
- [ ] Run the tests and verify failure because `training.feature_cache` does
  not exist.
- [ ] Implement dtype range checks, component buffering, 64-byte-aligned
  `data.bin`, and section metadata in `meta.json`.
- [ ] Implement mmap section views and physical file-size validation.
- [ ] Run the feature-cache tests and verify they pass.

### Task 2: Cache construction CLI

**Files:**
- Create: `imitation_learning/training/cache_features.py`
- Create: `imitation_learning/cfg/cache.yaml`
- Extend: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- `CacheSettings(cg_path, input, output, workers, samples_per_shard, force)`.
- `prepare_record(record, model_config) -> FeatureRecord | None`.
- `process_source(job) -> dict` writes `<source>.part-xxxxx.cache`.

- [ ] Write failing tests for sample-count part boundaries and resumable
  metadata validation.
- [ ] Implement YAML loading and early `cg_path` bootstrapping.
- [ ] Reuse `model.features.encoder_features`, `decoder_features`, and
  `enumerate_actions` to build cache records.
- [ ] Process independent JSONL files in a process pool.
- [ ] Write source manifests last and rebuild stale/incomplete sources.
- [ ] Run cache construction tests.

### Task 3: Mmap dataset and shuffle modes

**Files:**
- Extend: `imitation_learning/training/feature_cache.py`
- Extend: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- `MmapFeatureDataset(root, expected_signature)`.
- `iter_batches(batch_size, shuffle_mode, seed, max_samples)`.
- `shuffle_mode` accepts only `"global"` or `"shard"`.

- [ ] Write failing tests proving global shuffle visits every global sample
  exactly once and shard shuffle visits every shard/sample exactly once.
- [ ] Implement global uint32 permutations below 2^32 and uint64 otherwise.
- [ ] Implement cumulative-count mapping from global IDs to shard/local IDs.
- [ ] Implement shard-order and shard-local permutations.
- [ ] Implement batch collation with 24 encoder words and 64 padded decoder
  action words, returning int32 indices/offsets and FP16 encoder values.
- [ ] Run dataset/shuffle tests.

### Task 4: Decoder implicit weights and mixed-precision helpers

**Files:**
- Modify: `imitation_learning/model/network.py`
- Modify: `imitation_learning/training/train.py`
- Create: `imitation_learning/tests/test_mixed_precision.py`

**Interfaces:**
- Network forward accepts decoder offsets without a decoder value tensor while
  remaining compatible with the old six-argument call.
- `PrecisionContext(name, device)` validates and owns autocast/scaler behavior.
- `train_batch(...) -> BatchResult` reports loss, accuracy, grad norm, scale,
  and whether the optimizer step ran.

- [ ] Write a failing test comparing decoder output with explicit all-one
  weights versus omitted weights.
- [ ] Write failing configuration tests for `fp32`, `fp16`, `bf16`, invalid
  precision, and unsupported CPU mixed precision.
- [ ] Make decoder per-sample weights optional.
- [ ] Implement FP16 autocast plus GradScaler and BF16 autocast without scaler.
- [ ] Unscale FP16 gradients before clipping.
- [ ] Advance the scheduler only when GradScaler executes the optimizer step.
- [ ] Run mixed-precision tests; CUDA-only smoke cases skip when unsupported.

### Task 5: Replace preloading training loop

**Files:**
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/cfg/train.yaml`
- Extend: `imitation_learning/tests/test_mixed_precision.py`

**Interfaces:**
- Train settings add `precision` and `shuffle_mode`.
- Train settings remove `preload`, `preload_workers`,
  `preload_chunk_size`, and `shuffle_buffer`.
- Training consumes `MmapFeatureDataset.iter_batches`.

- [ ] Write failing settings tests for the new YAML fields.
- [ ] Remove raw JSON/preload code paths from training.
- [ ] Validate cache feature signature against current card/attack constants.
- [ ] Use cache sample totals for epoch and scheduler step counts.
- [ ] Log precision, gradient scale, skipped updates, cache samples, and shuffle
  mode to console/W&B.
- [ ] Preserve checkpoint payloads and add cache/precision metadata.
- [ ] Run the complete test suite.

### Task 6: Dependencies, documentation, and verification

**Files:**
- Create: `imitation_learning/requirements.txt`
- Modify: `imitation_learning/README.md`
- Modify: `docs/superpowers/specs/2026-07-25-mmap-cache-mixed-precision-design.md`

- [ ] Add explicit NumPy, PyYAML, PyTorch, W&B, and pytest dependencies.
- [ ] Document `extract -> cache_features -> train`.
- [ ] Document cache rebuild rules, shard size, shuffle modes, and precision
  choices.
- [ ] Run `python -m py_compile` for all changed Python modules.
- [ ] Run `pytest imitation_learning/tests -q`.
- [ ] Run a small cache build and one FP32 optimizer step.
- [ ] On supported CUDA, run one FP16 and one BF16 optimizer step.
- [ ] Review `git diff --check` and the final scoped diff.
