# Date-Weighted Training Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic linear or power date weighting that expands the final training sample index once at startup while preserving every validation split.

**Architecture:** A focused `training/date_sampling.py` module owns curve calculation and deterministic index expansion. `MmapFeatureDataset` supplies vectorized date labels for global indices, while `train.py` parses settings, builds the weighted index after `splits.train`, uses its length for scheduling, and reports realized sampling statistics.

**Tech Stack:** Python 3.10, NumPy, PyYAML, pytest, existing mmap feature cache and PyTorch training loop.

## Global Constraints

- Weight samples only after final `splits.train` construction.
- Apply identical date weights to winning and eligible losing samples.
- Sample fractional copies once at startup and globally shuffle the resulting fixed index each epoch.
- Use elapsed calendar days, not date-list rank.
- Do not change extraction, cache schema, model architecture, validation membership, or evaluation metrics.
- Preserve the user's existing `imitation_learning/deck/deck_eda.ipynb` modification.

---

### Task 1: Date curve and deterministic sample expansion

**Files:**
- Create: `imitation_learning/training/date_sampling.py`
- Create: `imitation_learning/tests/test_date_sampling.py`

**Interfaces:**
- Produces: `DateSamplingCurve`, `DateSamplingResult`, `date_weights(...)`, and `build_date_weighted_indices(...)` for Task 3.
- Consumes: one-dimensional global indices plus an aligned sequence of `(month, day)` dates.

- [ ] **Step 1: Write failing unit tests**

Cover exact linear endpoints/calendar midpoint, power exponent, single-date `end`, weights `0`, `0.5`, `1.2`, `2.5`, deterministic equal-seed output, different-seed fractional output, disabled identity behavior, and empty-result rejection. Use compact synthetic arrays so expected multiplicities can be checked with `np.bincount`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
$env:PYTHONPATH="$PWD\imitation_learning"
python -m pytest imitation_learning/tests/test_date_sampling.py -q
```

Expected: collection fails because `training.date_sampling` does not exist.

- [ ] **Step 3: Implement the sampling module**

Define immutable dataclasses:

```python
@dataclass(frozen=True, slots=True)
class DateSamplingCurve:
    mode: str
    start: float
    end: float
    exponent: float | None = None

@dataclass(frozen=True, slots=True)
class DateSamplingDateStats:
    date: tuple[int, int]
    coordinate: float
    weight: float
    source_samples: int
    weighted_samples: int

@dataclass(frozen=True, slots=True)
class DateSamplingResult:
    indices: np.ndarray
    dates: tuple[DateSamplingDateStats, ...]
```

Implement ordinal conversion with `datetime.date(2001, month, day).toordinal()`. Implement `date_weights` with one-date `end` behavior. Implement expansion using `floor(weight)` full copies plus `rng.random(count) < fraction`, preserving the input integer dtype. Return the input array itself when disabled so identity semantics are testable.

- [ ] **Step 4: Run unit tests**

Run the Task 1 pytest command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add imitation_learning/training/date_sampling.py imitation_learning/tests/test_date_sampling.py
git commit -m "feat: add deterministic date sampling"
```

### Task 2: Vectorized dataset date lookup

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- Produces: `MmapFeatureDataset.dates_for_indices(indices: np.ndarray) -> np.ndarray`, returning aligned integer month/day pairs with shape `[N, 2]`.
- Consumes: dataset `starts`, `ends`, and `shard_dates` already initialized from cache metadata.

- [ ] **Step 1: Write failing cache date-lookup tests**

Extend the existing packed-shard fixture to create at least two dated shards. Assert that unsorted global indices crossing shard boundaries return aligned dates, repeated indices repeat dates, and out-of-range or non-1D inputs raise clear errors.

- [ ] **Step 2: Run focused test and verify failure**

Run:

```powershell
$env:PYTHONPATH="$PWD\imitation_learning"
python -m pytest imitation_learning/tests/test_feature_cache.py -q
```

Expected: failure because `dates_for_indices` is absent.

- [ ] **Step 3: Implement vectorized lookup**

Convert indices to `int64`, validate dimensionality and bounds, find shards with `np.searchsorted(self.ends, indices, side="right")`, index an `np.asarray(self.shard_dates, dtype=np.int16)` lookup table, and return the resulting `[N, 2]` array. Do not call `sample()` or touch feature arrays.

- [ ] **Step 4: Run focused test**

Run the Task 2 pytest command. Expected: all feature-cache tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add imitation_learning/training/feature_cache.py imitation_learning/tests/test_feature_cache.py
git commit -m "feat: expose cache sample dates"
```

### Task 3: YAML settings, training integration, and reporting

**Files:**
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/training/train.py`
- Create: `imitation_learning/tests/test_date_sampling_integration.py`

**Interfaces:**
- Consumes: Task 1 `DateSamplingCurve` and `build_date_weighted_indices`; Task 2 `dates_for_indices`.
- Produces: validated `DateSamplingSettings` nested under `TrainSettings`, weighted indices used by scheduler and epoch iterator, console/W&B sampling metrics.

- [ ] **Step 1: Write failing settings and integration tests**

Create a temporary YAML derived from the project config and test parsing of `enabled`, `mode`, `seed`, both curve mappings, invalid modes, negative/non-finite weights, and non-positive/non-finite exponent. Add a pure integration helper test asserting weighted length controls `samples_per_epoch`/`steps_per_epoch` while the supplied validation indices are not passed to the sampler.

- [ ] **Step 2: Run integration tests and verify failure**

Run:

```powershell
$env:PYTHONPATH="$PWD\imitation_learning;$PWD\pokemon_tcg_ai_battle\sample_submission"
python -m pytest imitation_learning/tests/test_date_sampling_integration.py -q
```

Expected: failure because date-sampling settings and integration helpers are absent.

- [ ] **Step 3: Add configuration parsing and validation**

Add frozen settings dataclasses for linear/power curves and the enclosing date sampling mapping. Pop `date_sampling` from `train_raw`, construct the nested settings explicitly, then validate booleans/types, mode membership, finite non-negative endpoints, positive finite exponent, and integer seed. Add the confirmed YAML block with concise comments.

- [ ] **Step 4: Integrate weighted indices after split construction**

After `dataset.build_splits(...)`, call `dataset.dates_for_indices(splits.train)` and `build_date_weighted_indices`. Store the result as `train_indices`. Derive `samples_per_epoch` and all scheduler counts from `len(train_indices)`, then pass `train_indices` rather than `splits.train` to `dataset.iter_batches`. Keep `max_samples` after weighting.

- [ ] **Step 5: Add startup and W&B reporting**

Print the specified per-date and aggregate lines. Add aggregate keys and sanitized per-date keys such as `data/date_sampling/7_1_weight` and `data/date_sampling/7_1_weighted_samples` to the existing initialization payload. Preserve existing `data/train_samples` as the unweighted split count and add `data/weighted_train_samples`.

- [ ] **Step 6: Run focused integration tests**

Run the Task 3 pytest command. Expected: all tests pass.

- [ ] **Step 7: Run related regression tests**

Run:

```powershell
$env:PYTHONPATH="$PWD\imitation_learning;$PWD\pokemon_tcg_ai_battle\sample_submission"
python -m pytest imitation_learning/tests/test_date_sampling.py imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_training_schedule.py imitation_learning/tests/test_training_metrics.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 3**

```powershell
git add imitation_learning/cfg/train.yaml imitation_learning/training/train.py imitation_learning/tests/test_date_sampling_integration.py
git commit -m "feat: weight training samples by date"
```

### Task 4: Final compatibility verification

**Files:**
- Verify only; no planned production edits.

**Interfaces:**
- Consumes: the completed date-sampling feature.
- Produces: evidence that extraction/cache/model/validation compatibility is unchanged.

- [ ] **Step 1: Compile modified Python sources**

Compile `date_sampling.py`, `feature_cache.py`, and `train.py` using the configured Conda Python without writing application data.

- [ ] **Step 2: Run the focused regression suite**

Run all tests from Tasks 1–3 and confirm zero failures.

- [ ] **Step 3: Verify repository scope**

Run `git diff --check`, inspect `git status --short`, and confirm `imitation_learning/deck/deck_eda.ipynb` remains the only unrelated user modification and is not included in feature commits.

- [ ] **Step 4: Document migration in the handoff**

Report that neither extraction nor cache rebuild is required, validation membership is unchanged, and weights are reconstructed deterministically at startup/resume.
