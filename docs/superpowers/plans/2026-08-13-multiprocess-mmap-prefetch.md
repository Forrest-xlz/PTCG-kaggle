# Multiprocess mmap Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare cached training batches in multiple CPU processes while preserving the existing deterministic sample and batch order.

**Architecture:** The main process remains responsible for seeded index generation and all GPU work. A focused prefetch module owns synchronous, threaded, and spawned-process collation; each process opens the mmap dataset once and results are yielded in submission order through a bounded future queue.

**Tech Stack:** Python 3.10, NumPy mmap, `concurrent.futures.ProcessPoolExecutor`, PyTorch, pytest, YAML.

## Global Constraints

- Do not change extraction, cache schema, split logic, model inputs, model architecture, losses, validation, or checkpoint formats.
- Use the multiprocessing `spawn` context and never initialize CUDA in a worker.
- Bound in-flight work by `train.prefetch_batches` and preserve exact `IndexBatch` ordering.
- Keep `prefetch_workers: 0` synchronous and `prefetch_workers: 1` compatible with the existing background-thread behavior.

---

### Task 1: Ordered batch prefetch module

**Files:**
- Create: `imitation_learning/training/batch_prefetch.py`
- Create: `imitation_learning/tests/test_batch_prefetch.py`
- Modify: `imitation_learning/tests/test_multitask_training.py`

**Interfaces:**
- Consumes: `MmapFeatureDataset`, `IndexBatch`, and its existing `collate` method.
- Produces: `iter_collated_batches(dataset, index_batches, workers, buffer_size, expected_signature)` yielding `CachedBatch` objects in input order.

- [ ] **Step 1: Write failing tests for synchronous, threaded, and process order plus worker exception propagation**

```python
batch_ids = [IndexBatch(np.array([2])), IndexBatch(np.array([0, 1]))]
markers = [batch.own_summary[:, 0].tolist() for batch in iter_collated_batches(...)]
assert markers == [[3.0], [1.0, 2.0]]
```

- [ ] **Step 2: Run the focused tests and confirm import failure because the module does not exist**

Run: `python -m pytest imitation_learning/tests/test_batch_prefetch.py -q`
Expected: FAIL with `ModuleNotFoundError: training.batch_prefetch`.

- [ ] **Step 3: Implement bounded ordered collation**

```python
def iter_collated_batches(dataset, index_batches, *, workers, buffer_size, expected_signature):
    if workers == 0 or buffer_size == 0:
        yield from (dataset.collate(batch) for batch in index_batches)
    elif workers == 1:
        yield from prefetch_iterable((dataset.collate(batch) for batch in index_batches), buffer_size)
    else:
        yield from _iter_process_batches(dataset.root, expected_signature, index_batches, workers, buffer_size)
```

- [ ] **Step 4: Run focused tests and confirm all modes pass**

Run: `python -m pytest imitation_learning/tests/test_batch_prefetch.py imitation_learning/tests/test_multitask_training.py -q`
Expected: PASS.

### Task 2: Training and YAML integration

**Files:**
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/tests/test_batch_prefetch.py`

**Interfaces:**
- Consumes: `train.prefetch_workers`, `train.prefetch_batches`, and the current model feature signature.
- Produces: training batches through `iter_collated_batches` without changing the downstream `train_batch` call.

- [ ] **Step 1: Add a failing configuration validation test**

```python
assert validate_prefetch_settings(workers=-1, batches=8) raises ValueError
```

- [ ] **Step 2: Run it and confirm failure because worker validation is absent**

Run: `python -m pytest imitation_learning/tests/test_batch_prefetch.py -q`
Expected: FAIL for the new worker-validation assertion.

- [ ] **Step 3: Add configuration, validation, and training-loop integration**

```yaml
prefetch_workers: 4
prefetch_batches: 8
```

Replace `prefetch_iterable(dataset.iter_batches(...))` with main-process `iter_index_batches(...)` passed to `iter_collated_batches(...)`.

- [ ] **Step 4: Run focused and full regression verification**

Run: `python -m pytest imitation_learning/tests -q`
Expected: all tests pass.

- [ ] **Step 5: Compile modified modules and inspect the final diff**

Run: `python -m py_compile imitation_learning/training/batch_prefetch.py imitation_learning/training/train.py`
Expected: exit code 0.

