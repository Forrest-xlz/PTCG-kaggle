# Isolation Validation Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the three reviewed deck-selection CSVs into replay-level validation sets, exclude their union before latest-date and in-distribution splitting, and log three independent Wandb metric groups from one isolation-union forward pass.

**Architecture:** A focused `training/isolation_validation.py` module converts selected exact decks plus two-player deck extracts into per-date episode-key sets. `MmapFeatureDataset.build_splits()` applies those sets before every existing split and exposes one isolation index array with three aligned masks. `training/train.py` owns typed YAML configuration, startup auditing, and independent metric logging.

**Tech Stack:** Python 3, pandas, NumPy, PyTorch, PyYAML, pytest, Wandb.

## Global Constraints

- Do not change the packed feature-cache schema or require cache rebuilding.
- A replay matches an isolation set when either player's exact deck is selected.
- The three isolation sets may overlap with one another.
- Their union must not overlap latest-date validation, in-distribution validation, or training.
- Latest-date validation is built after removing the isolation union.
- In-distribution validation is sampled only from remaining older-date replays.
- Evaluate the isolation union once and publish only three separate metric namespaces.
- Do not create a `val_isolation/*` Wandb metric group.

---

### Task 1: Resolve Selection Decks to Replay Sets

**Files:**
- Create: `imitation_learning/training/isolation_validation.py`
- Create: `imitation_learning/tests/test_isolation_validation.py`

**Interfaces:**
- Consumes: `stable_deck_key()` and `stable_episode_key()` from `training.feature_cache`.
- Produces: `IsolationReplaySets`, containing `by_namespace: dict[str, dict[tuple[int, int], frozenset[int]]]`, selected-deck counts, replay counts, union replay count, and pairwise overlap counts.
- Produces: `load_isolation_replay_sets(deck_data_dir: Path, selection_paths: Mapping[str, Path], required_dates: Iterable[tuple[int, int]]) -> IsolationReplaySets`.

- [ ] **Step 1: Add failing parser and matching tests**

Create fixtures with two selection CSVs sharing one replay and a third disjoint
selection. Assert that a selected opponent deck isolates the entire replay,
overlap is retained in both namespaces, all required dates exist in the
returned mappings, and counts are deduplicated by `(date, episode_id)`.

- [ ] **Step 2: Add failing validation tests**

Cover a missing `card_ids` column, malformed JSON, a list other than 60
non-negative integers, a selected deck absent from deck extracts, a missing
deck-data directory, and an isolation namespace resolving to no required-date
replay.

- [ ] **Step 3: Implement strict selection parsing**

Read each CSV with pandas, parse `card_ids` with `json.loads`, validate through
`stable_deck_key`, collapse duplicate deck keys, and retain `deck_id` when
present for diagnostics.

- [ ] **Step 4: Implement one-pass deck-data scanning**

Read sorted `*.decks.csv` files, require `date`, `episode_id`, and `deck`,
parse each exact deck, and add `stable_episode_key(episode_id)` to every
matching namespace/date. Count replay identities as `(date_tuple,
episode_key)` and calculate all pairwise overlaps.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
pytest imitation_learning/tests/test_isolation_validation.py -q
```

Expected: all resolver tests pass.

### Task 2: Apply Isolation Before Existing Dataset Splits

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- Consumes: `isolation_episode_keys: Mapping[str, Mapping[tuple[int, int], AbstractSet[int]]]`.
- Extends: `DatasetSplits` with `isolation`, `isolation_masks`, `isolation_sample_counts`, and `isolation_union_replays`.
- Preserves: existing train, in-distribution, latest-date, expert, and top-deck fields.

- [ ] **Step 1: Add a failing split-precedence test**

Construct old-date and latest-date cache shards containing replay keys assigned
to overlapping isolation namespaces. Assert that the isolation masks align
with the union, overlapping samples have both masks true, and the isolation
union is disjoint from latest, in-distribution, and train.

- [ ] **Step 2: Add failing latest and ratio tests**

Assert that isolation samples on the numerically latest date are absent from
`latest`; in-distribution hash selection sees only non-isolation older
replays; and `train_replay_ratio` sees only the remainder.

- [ ] **Step 3: Extend the split result**

Add one sorted isolation-union global-index array and a dictionary of aligned
boolean masks. Keep the low-level parameter optional for existing direct
callers, but require nonempty namespaces whenever a mapping is supplied.

- [ ] **Step 4: Implement shard-local isolation masks**

For each shard/date, build namespace masks with `np.isin`, combine them with
logical OR, append union indices and namespace masks restricted to the union,
then apply latest-date, in-distribution, and train logic only to the inverse
union.

- [ ] **Step 5: Add invariant checks**

Reject missing required dates, empty configured subsets, an empty latest set
after isolation, and any intersection between the isolation union and later
splits.

- [ ] **Step 6: Run focused cache tests**

Run:

```powershell
pytest imitation_learning/tests/test_feature_cache.py -q
```

Expected: all existing and new cache tests pass without rebuilding project
cache artifacts.

### Task 3: Integrate Typed Configuration and Training Evaluation

**Files:**
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/tests/test_artifacts.py`
- Create: `imitation_learning/tests/test_train.py`

**Interfaces:**
- Produces: `IsolationValidationSettings(deck_data: str, selections: dict[str, str])`.
- Consumes: `load_isolation_replay_sets()` and the extended `DatasetSplits`.
- Publishes: `val_deck_isolation/*`, `val_archetype_isolation/*`, and `val_top_deck_archetype_isolation/*`.

- [ ] **Step 1: Add failing YAML and settings tests**

Require the nested `train.isolation_validation` mapping, the exact three
selection keys, nonempty path strings, and successful interpolation/loading
into `IsolationValidationSettings`.

- [ ] **Step 2: Add failing validation-loop behavior tests**

Extract a helper that evaluates the isolation union with three subgroup masks.
Use a stub evaluator/logger to assert one dataset evaluation and three
independent `_log_validation` calls, with no `val_isolation` log.

- [ ] **Step 3: Add YAML configuration**

Configure:

```yaml
isolation_validation:
  deck_data: data/deck
  selections:
    deck_isolation: data/deck_isolation_selection.csv
    archetype_isolation: data/archetype_isolation_selection.csv
    top_deck_archetype_isolation: data/top_deck_archetype_isolation_selection.csv
```

- [ ] **Step 4: Implement typed settings parsing**

Pop the nested mapping before constructing `TrainSettings`, construct
`IsolationValidationSettings`, validate the exact selection-key set, and keep
`asdict()` checkpoint/Wandb configuration serialization working.

- [ ] **Step 5: Resolve isolation replay sets before splitting**

After opening the mmap dataset, resolve paths relative to `PROJECT_ROOT`, load
the replay sets for `dataset.shard_dates`, print selection diagnostics, and
pass `by_namespace` into `build_splits()`.

- [ ] **Step 6: Register independent Wandb groups**

Add only:

```text
val_deck_isolation/*
val_archetype_isolation/*
val_top_deck_archetype_isolation/*
```

to metric definitions, each stepped by `optimizer_step`.

- [ ] **Step 7: Evaluate the isolation union once**

Call `evaluate_dataset()` once with `splits.isolation` and
`splits.isolation_masks`. Ignore the union's overall model metrics and call
`_log_validation()` separately for each subgroup result. Continue with
latest-date and in-distribution evaluations afterward.

- [ ] **Step 8: Add startup and Wandb data audits**

Print and log per-isolation replay/sample counts, union counts, pairwise
overlaps, and post-isolation latest, in-distribution, and training counts.

- [ ] **Step 9: Run focused training/config tests**

Run:

```powershell
pytest imitation_learning/tests/test_train.py imitation_learning/tests/test_artifacts.py -q
```

Expected: settings, namespace, and one-forward behavior tests pass.

### Task 4: Documentation and Full Verification

**Files:**
- Modify: `imitation_learning/README.md`

**Interfaces:**
- Documents the required input paths, split precedence, overlap semantics, and Wandb groups.

- [ ] **Step 1: Update the training-data documentation**

Explain the exact split order, that either player's deck triggers isolation,
that the three isolation sets may overlap, and that no cache rebuild is
needed.

- [ ] **Step 2: Run the complete project tests**

Run:

```powershell
pytest imitation_learning/tests -q
```

Expected: all tests pass.

- [ ] **Step 3: Run static checks**

Run:

```powershell
git diff --check
```

Parse `cfg/train.yaml`, verify the three selection paths exist in the current
workspace, and confirm no `val_isolation/*` namespace occurs in training code.

- [ ] **Step 4: Review the scoped diff**

Confirm no feature-cache schema or extraction code changed and report the
test count, static-check results, and any environment warnings.
