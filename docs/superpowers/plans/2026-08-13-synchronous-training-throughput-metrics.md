# Synchronous Training Throughput Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore synchronous collation defaults and report separate data, compute, and end-to-end training throughput.

**Architecture:** Add one pure throughput-calculation helper in `training/train.py`, use it in the existing log block, and change only the checked-in prefetch defaults. Optional thread and process prefetch implementations remain intact.

**Tech Stack:** Python 3.10, PyTorch, YAML, pytest.

## Global Constraints

- Do not change the model, auxiliary losses, cache schema, splits, validation, optimizer, or checkpoint payload.
- Keep `train/samples_per_second` as an end-to-end compatibility alias.
- Accumulate timing across the complete run rather than resetting at epoch boundaries.

---

### Task 1: Throughput calculation and logging

**Files:**
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/tests/test_training_metrics.py`

**Interfaces:**
- Consumes: cumulative sample count, data seconds, and compute seconds.
- Produces: `training_throughput_metrics(samples, data_seconds, compute_seconds)` with data, compute, end-to-end rates and data-wait ratio.

- [ ] **Step 1: Write failing tests for normal and zero-data-time calculations**

```python
metrics = training_throughput_metrics(1000, 0.25, 0.75)
assert metrics["compute_samples_per_second"] == 1000 / 0.75
assert metrics["end_to_end_samples_per_second"] == 1000
assert metrics["samples_per_second"] == 1000
assert metrics["data_wait_ratio"] == 0.25
```

- [ ] **Step 2: Run the focused test and observe the missing-helper failure**

Run: `pytest imitation_learning/tests/test_training_metrics.py -q`
Expected: FAIL because `training_throughput_metrics` does not exist.

- [ ] **Step 3: Implement the pure helper and replace inline throughput arithmetic**

```python
def training_throughput_metrics(samples, data_seconds, compute_seconds):
    total = data_seconds + compute_seconds
    end_to_end = samples / max(total, 1e-9)
    return {
        "data_samples_per_second": samples / data_seconds if data_seconds > 0 else 0.0,
        "compute_samples_per_second": samples / max(compute_seconds, 1e-9),
        "end_to_end_samples_per_second": end_to_end,
        "samples_per_second": end_to_end,
        "data_wait_ratio": data_seconds / max(total, 1e-9),
    }
```

- [ ] **Step 4: Run focused tests and confirm they pass**

Run: `pytest imitation_learning/tests/test_training_metrics.py -q`
Expected: PASS.

### Task 2: Synchronous defaults and regression verification

**Files:**
- Modify: `imitation_learning/cfg/train.yaml`
- Test: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- Consumes: existing `prefetch_workers` and `prefetch_batches` settings.
- Produces: checked-in defaults `0` and `0`, selecting direct `dataset.collate` execution.

- [ ] **Step 1: Change the two YAML defaults to zero with concise comments**

```yaml
prefetch_workers: 0
prefetch_batches: 0
```

- [ ] **Step 2: Run relevant tests, compile modules, and load the checked-in config**

Run: `pytest imitation_learning/tests/test_training_metrics.py imitation_learning/tests/test_feature_cache.py -q`
Expected: PASS, followed by a config assertion that both values are zero.

- [ ] **Step 3: Commit the implementation**

```bash
git add imitation_learning/cfg/train.yaml imitation_learning/training/train.py imitation_learning/tests/test_training_metrics.py
git commit -m "perf: restore synchronous training input"
```
