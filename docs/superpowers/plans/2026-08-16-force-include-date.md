# Force-Include Date Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train on every cached player sample from configured dates such as `8.16`, while keeping all other dates winner-only and removing full-training manifest/loser augmentation.

**Architecture:** Extend full-training index construction with an explicit set of force-included dates without changing packed cache contents or schema. Parse that set from `train.yaml`; the training entry point passes it to the dataset and no longer loads expert-loser manifests.

**Tech Stack:** Python 3.10+, NumPy, PyYAML, pytest, PyTorch

## Global Constraints

- Keep extraction schema at 4 and cache schema at 16.
- Do not change model architecture or feature tensors.
- Preserve `imitation_learning/eda/deck_trend.ipynb` unchanged.
- Do not implement date/team oversampling.

---

### Task 1: Force-Included Date Selection

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- Consumes: existing `MmapFeatureDataset.build_training_indices(...)`
- Produces: optional `include_all_dates: AbstractSet[tuple[int, int]] | None` argument

- [ ] **Step 1: Write failing selection tests**

Add a cache fixture containing win/loss/draw samples on `8.16` and a win/loss pair on `8.15`. Assert that `include_all_dates={(8, 16)}` selects all three `8.16` samples but only the `8.15` winner. Add an assertion that `(8, 17)` raises `ValueError` because it is absent from cache.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
python -m pytest imitation_learning/tests/test_feature_cache.py -q
```

Expected: failure because `build_training_indices` does not accept `include_all_dates`.

- [ ] **Step 3: Implement the minimal selection rule**

Validate configured dates against `self.shard_dates`. For an included date, use an all-true eligibility mask; otherwise use `player_result == PLAYER_RESULT_WIN`. Keep replay-ratio selection unchanged.

- [ ] **Step 4: Run the focused test and verify pass**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add imitation_learning/training/feature_cache.py imitation_learning/tests/test_feature_cache.py
git commit -m "feat: force include configured training dates"
```

### Task 2: Remove Full-Training Loser Augmentation

**Files:**
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/tests/test_training_metrics.py`

**Interfaces:**
- Consumes: YAML `train.include_all_dates: list[str]`
- Produces: parsed `TrainSettings.include_all_dates: tuple[str, ...]`, passed as parsed `(month, day)` tuples to Task 1

- [ ] **Step 1: Write failing configuration tests**

Assert that the repository YAML contains `include_all_dates: ["8.16"]`, that `load_settings()` preserves it, and that invalid date labels fail through `parse_source_date`. Update loser-date tests so the full-training entry point no longer requires loser augmentation configuration.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
python -m pytest imitation_learning/tests/test_training_metrics.py -q
```

Expected: failure because `TrainSettings` has no `include_all_dates` setting and still requires loser augmentation.

- [ ] **Step 3: Implement configuration and training integration**

Remove `LoserAugmentationSettings`, expert-loser imports, manifest loading, loser reporting, and loser W&B metrics from `train.py`. Add and validate `include_all_dates`, parse labels with `parse_source_date`, and call:

```python
selection = dataset.build_training_indices(
    train_replay_ratio=train_cfg.train_replay_ratio,
    train_replay_seed=train_cfg.train_replay_seed,
    include_all_dates=include_all_dates,
)
```

Replace the YAML loser-augmentation mapping with:

```yaml
include_all_dates: ["8.16"]
```

- [ ] **Step 4: Run targeted verification**

Run the Task 1 and Task 2 test files, compile `feature_cache.py` and `train.py`, parse the repository YAML, and run `git diff --check`.

- [ ] **Step 5: Commit Task 2**

```powershell
git add imitation_learning/cfg/train.yaml imitation_learning/training/train.py imitation_learning/tests/test_training_metrics.py
git commit -m "feat: include manual replay date in full training"
```
