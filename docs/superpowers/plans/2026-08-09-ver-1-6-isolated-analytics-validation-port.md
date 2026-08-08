# Ver 1.6 Isolated Analytics and Validation Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ordered per-deck validation, deck trend EDA, and replay timing EDA to `ver_1.6.0` without changing the `ver_1.4.5` model, feature, cache-schema, history, or submission architecture.

**Architecture:** Replay timing and deck trend are transferred from their isolated commits because those commits contain only analytics files, dependency additions, tests, and documentation. Validation is ported hunk-by-hunk into the current `ver_1.4.5` training and split code so its activation, dropout, history, and cache layout remain untouched.

**Tech Stack:** Python, PyTorch, NumPy, pandas, scikit-learn, matplotlib, seaborn, Jupyter, pytest, WandB, Git.

## Global Constraints

- Keep `ver_1.6.0` based on commit `0420d38` (`ver_1.4.5`).
- Do not modify `model/features.py`, `model/network.py`, `training/cache_features.py`, the Kaggle submission notebook, or `cfg/train.yaml`.
- Do not increment `CACHE_SCHEMA_VERSION` or alter packed-cache storage/collation.
- Preserve transformer activation, four dropout switches, and action-history configuration and data flow.
- Per-deck numbering follows `train.top_decks` YAML order starting at one.
- Each WandB validation namespace contains only loss and top-1/3/5 accuracy.
- Latest and in-distribution subgroup metrics reuse their base validation forward passes.

---

### Task 1: Restore Replay Timing EDA Only

**Files:**
- Create: `imitation_learning/eda/replay_timing.ipynb`
- Modify: `imitation_learning/requirements.txt`
- Modify: `imitation_learning/README.md`

**Interfaces:**
- Consumes: latest dated replay archive/directory and configurable score threshold/mode in the notebook.
- Produces: team-level timing CSV under `imitation_learning/data` and timing distribution/K-means figures.

- [ ] **Step 1: Inspect the isolated source commit**

Run `git show --stat --oneline 528294c` and confirm it changes only README,
requirements, and `eda/replay_timing.ipynb`.

- [ ] **Step 2: Apply the isolated commit**

Run `git cherry-pick 528294c`. Expect no model, training, feature, cache,
configuration, or submission file changes.

- [ ] **Step 3: Verify the task boundary**

Run `git diff --name-only 0420d38..HEAD` and confirm only the three files above
and the approved design/plan documents appear for this task.

---

### Task 2: Add Deck Trend Extraction and EDA

**Files:**
- Create: `imitation_learning/cfg/deck_trend.yaml`
- Create: `imitation_learning/deck/trend.py`
- Create: `imitation_learning/deck/trend_extract.py`
- Create: `imitation_learning/eda/deck_trend.ipynb`
- Create: `imitation_learning/tests/test_deck_trend.py`
- Create: `imitation_learning/tests/test_deck_trend_extract.py`
- Create: `imitation_learning/tests/test_deck_trend_notebook.py`
- Modify: `imitation_learning/requirements.txt`
- Modify: `imitation_learning/README.md`

**Interfaces:**
- Consumes: replay archives configured by `cfg/deck_trend.yaml` and existing manual archetype rules.
- Produces: reusable deck-level tables and interval-based usage, win-rate, and matchup analysis.

- [ ] **Step 1: Inspect the source commit**

Run `git show --stat --oneline e482b89` and confirm it contains the deck trend
workflow, tests/docs, README, and one requirements addition only.

- [ ] **Step 2: Apply the isolated commit**

Run `git cherry-pick e482b89`. Expect no model or training-pipeline conflict.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
python -m pytest imitation_learning/tests/test_deck_trend.py imitation_learning/tests/test_deck_trend_extract.py imitation_learning/tests/test_deck_trend_notebook.py -q
```

Expected: all focused tests pass. If Python is unavailable, record the
constraint and use static Git checks without claiming tests passed.

---

### Task 3: Port Ordered Per-Deck Validation

**Files:**
- Modify: `imitation_learning/tests/test_feature_cache.py`
- Modify: `imitation_learning/tests/test_training_metrics.py`
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/README.md`

**Interfaces:**
- Consumes: ordered `tuple[int, ...]` of stable exact-deck keys from `train.top_decks`.
- Produces: per-deck masks on `DatasetSplits` and ordered mappings from `top_deck_subgroup_masks(scope, deck_masks, expert_deck_masks)`.

- [ ] **Step 1: Add failing split tests**

Extend the shard helper with aligned optional `deck_markers`. Build two dates
containing two exact decks and assert two ordered masks exist for each of
in-distribution, in-distribution expert, latest, and latest expert. Assert the
latest masks equal `[True, False, False]` for `deck1` and
`[False, True, False]` for `deck2`. Keep every action-history field in existing
`FeatureRecord` fixtures.

- [ ] **Step 2: Add failing namespace and payload tests**

Assert the helper returns keys in this order:

```python
[
    "val_in_distribution_deck1",
    "val_in_distribution_expert_deck1",
    "val_in_distribution_deck2",
    "val_in_distribution_expert_deck2",
]
```

Assert `_log_validation` sends only loss, top-1/3/5 accuracy under its
validation namespace plus global `optimizer_step`.

- [ ] **Step 3: Run tests to confirm the missing behavior**

Run:

```powershell
python -m pytest imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_training_metrics.py -q
```

Expected: failures identify missing ordered mask fields and subgroup helper.

- [ ] **Step 4: Add runtime masks without changing cache storage**

Add ordered per-deck and expert-intersection mask tuples to `DatasetSplits`.
Make `build_splits(top_deck_keys=...)` preserve iterable order, create one mask
per key per shard, concatenate masks in split order, and retain existing union
masks. Reject empty configured `deckN` subgroups. Do not edit schema constants,
packed sections, sample records, or collation.

- [ ] **Step 5: Add ordered training namespaces**

Keep `top_deck_keys` as a tuple and reject duplicates. Add
`top_deck_subgroup_masks`, which interleaves ordinary and expert groups for
each one-based deck index. Pass these masks to the existing latest and
in-distribution `evaluate_dataset` calls; do not add forward passes. Define all
four `deckN` namespace families dynamically. Preserve existing overall and
overall expert groups. Move sample counts to `data/*`; validation groups log
only loss and top-1/3/5 accuracy.

- [ ] **Step 6: Run focused validation tests**

Run:

```powershell
python -m pytest imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_training_metrics.py -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit validation changes**

Stage only the five Task 3 files and commit with
`feat: add ordered per-deck validation metrics`.

---

### Task 4: Verify Architectural Isolation

**Files:**
- Verify only; no production files change.

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: evidence that analytics and validation did not regress the architecture.

- [ ] **Step 1: Compare protected files to the baseline**

Run:

```powershell
git diff --exit-code 0420d38..HEAD -- imitation_learning/model/features.py imitation_learning/model/network.py imitation_learning/training/cache_features.py imitation_learning/kaggle_submission_imitation_agent.ipynb imitation_learning/cfg/train.yaml
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Check architecture wiring remains present**

Run:

```powershell
rg "transformer_activation|dropout_embedding|history_encoding" imitation_learning/training/train.py imitation_learning/model/network.py imitation_learning/cfg/train.yaml
```

Expected: fields remain in YAML, settings/configuration, and model construction.

- [ ] **Step 3: Run the complete suite**

Run `python -m pytest imitation_learning/tests -q`. Expected: all tests pass.
If Python is unavailable, report that tests were not run.

- [ ] **Step 4: Validate the final patch**

Run `git diff --check 0420d38..HEAD` and `git status --short`. Expect no
whitespace errors and a clean working tree.
