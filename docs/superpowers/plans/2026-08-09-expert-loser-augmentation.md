# Expert Loser Replay Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train on all winner actions plus losing-player actions from recent daily high-score replays while keeping every validation set winner-only and replay-isolated.

**Architecture:** Extraction stores both players with an explicit win/loss/draw result, and packed mmap caches persist that result without changing model inputs. Training loads daily manifest thresholds, builds existing replay-level validation splits first, and then admits only score-qualified loss samples from the most recent non-latest training dates. Split accounting is returned to `train.py` for terminal and WandB reporting.

**Tech Stack:** Python 3.10+, PyYAML, NumPy mmap caches, PyTorch training, pytest, WandB.

## Global Constraints

- Daily score cutoffs are calculated independently with `ceil(participant_count * expert_ratio)` and include ties via `min_score >= cutoff`.
- `recent_dates` counts the newest cached dates after excluding the latest-date validation archive.
- All validation datasets remain winner-only; no replay assigned to any validation split may contribute either player's training samples.
- `train_replay_ratio` selects whole replays consistently for winner and qualified loser samples.
- Draw samples never enter training or validation.
- Changing loser-augmentation settings after the schema upgrade must not require extraction or cache rebuilding.
- Do not modify model architecture, encoded feature values, optimization, checkpoint format, or validation metrics.

---

### Task 1: Daily Expert-Loser Manifest Selection

**Files:**
- Modify: `imitation_learning/training/expert_validation.py`
- Test: `imitation_learning/tests/test_expert_validation.py`

**Interfaces:**
- Produces: `ExpertLoserDateInfo(date, cutoff, participant_count, episode_count, eligible_episode_keys)`.
- Produces: `load_expert_loser_date_info(replay_root: Path, required_dates: Iterable[tuple[int, int]], ratio: float) -> dict[tuple[int, int], ExpertLoserDateInfo]`.
- Preserves: existing `load_expert_date_info` behavior for expert validation masks.

- [ ] **Step 1: Write failing manifest-selection tests**

Create ZIP fixtures containing `manifest.csv` and assert that each date computes its own participant-score cutoff, includes cutoff ties, and marks an episode only when its `min_score` reaches the cutoff. Also assert invalid ratios and missing dates retain clear errors.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest imitation_learning/tests/test_expert_validation.py -q`

Expected: FAIL because `ExpertLoserDateInfo` and `load_expert_loser_date_info` do not exist.

- [ ] **Step 3: Refactor manifest parsing and implement loser selection**

Parse each manifest row once into its stable episode key, low score, and high score. Keep expert-validation selection based on `high_score`; add loser selection based on `low_score`. Use the same archive/date validation and finite-score checks for both public loaders.

- [ ] **Step 4: Run the focused tests**

Run: `pytest imitation_learning/tests/test_expert_validation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the manifest selector**

```bash
git add imitation_learning/training/expert_validation.py imitation_learning/tests/test_expert_validation.py
git commit -m "feat: select daily expert loser replays"
```

### Task 2: Extract Both Players with Explicit Results

**Files:**
- Modify: `imitation_learning/training/extract.py`
- Modify: `imitation_learning/cfg/extract.yaml`
- Modify: `imitation_learning/tests/test_extract_alignment.py`

**Interfaces:**
- Produces JSONL field: `player_result: "win" | "loss" | "draw"` on every action record.
- Produces metadata: upgraded extraction schema and `player_results: "win-loss-draw"`.
- Removes the fixed-data behavior switch `extract.winner_only`; both players are always extracted.

- [ ] **Step 1: Write failing extraction tests**

Extend alignment tests to assert `_iter_player_records` writes the supplied result. Add small replay archive tests showing a decisive replay emits both `win` and `loss`, while a no-winner replay emits `draw` rather than being discarded.

- [ ] **Step 2: Run extraction tests and verify failure**

Run: `pytest imitation_learning/tests/test_extract_alignment.py -q`

Expected: FAIL because records do not contain `player_result` and extraction still filters to winners.

- [ ] **Step 3: Implement result classification and schema upgrade**

Classify all positive-reward players as winners and all other players as losers when a winner exists; when no positive reward exists, classify all players as draws. Always call `_iter_player_records` for every available player/deck and pass the result. Remove `winner_only` from `ExtractSettings`, YAML, job tuples, cache-reuse checks, and startup logging.

- [ ] **Step 4: Run extraction tests**

Run: `pytest imitation_learning/tests/test_extract_alignment.py -q`

Expected: PASS.

- [ ] **Step 5: Commit extraction changes**

```bash
git add imitation_learning/training/extract.py imitation_learning/cfg/extract.yaml imitation_learning/tests/test_extract_alignment.py
git commit -m "feat: extract both replay players"
```

### Task 3: Persist Player Results and Build Leakage-Safe Splits

**Files:**
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/tests/test_feature_cache.py`
- Modify: `imitation_learning/tests/test_history_feature_cache.py`
- Modify: `imitation_learning/tests/test_option_cache.py`

**Interfaces:**
- Produces constants: `PLAYER_RESULT_WIN`, `PLAYER_RESULT_LOSS`, and `PLAYER_RESULT_DRAW` as uint8-compatible codes.
- Extends `FeatureRecord` and packed sections with `player_result`.
- Extends `MmapFeatureDataset.build_splits(..., loser_episode_keys: Mapping[date, AbstractSet[int]] | None)`.
- Extends `DatasetSplits` with per-date `loser_augmentation_counts`, aggregate `loser_augmentation_replays`, `loser_augmentation_samples`, and `loser_fraction_in_train`.

- [ ] **Step 1: Write failing cache round-trip and split tests**

Add tests that round-trip all three player-result codes. Build synthetic shards containing winner/loss/draw records for the same replay keys and assert: validations contain winners only; validation replay losses never enter training; eligible selected replay losses do enter training; draws never enter any split; and replay-ratio selection treats both sides identically. Assert per-date and aggregate counts match the selected indices.

- [ ] **Step 2: Run focused cache tests and verify failure**

Run: `pytest imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_history_feature_cache.py imitation_learning/tests/test_option_cache.py -q`

Expected: FAIL because player results and loser episode sets are unsupported.

- [ ] **Step 3: Upgrade packed storage**

Increment `CACHE_SCHEMA_VERSION`, add a uint8 `player_result` section, validate result codes in the writer, expose the value in `FeatureView`, and map JSON result strings to codes in `_prepare_record`. Replace `_source_meta`'s `winner_only=true` requirement with the upgraded extraction schema/result marker and a rebuild-oriented error.

- [ ] **Step 4: Implement winner-only validation and loser-augmented training masks**

In `build_splits`, keep separate replay masks and sample masks. Use replay masks to prevent leakage and winner masks to populate validation arrays. Build training masks as selected replay AND (`win` OR (`loss` AND date-qualified episode)); exclude draws. Calculate unique replay and sample counts after validation exclusion and replay sampling for each augmentation date.

- [ ] **Step 5: Run focused cache tests**

Run: `pytest imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_history_feature_cache.py imitation_learning/tests/test_option_cache.py -q`

Expected: PASS.

- [ ] **Step 6: Commit cache and split changes**

```bash
git add imitation_learning/training/cache_features.py imitation_learning/training/feature_cache.py imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_history_feature_cache.py imitation_learning/tests/test_option_cache.py
git commit -m "feat: cache outcomes and augment train splits"
```

### Task 4: Training Configuration, Date Selection, and Reporting

**Files:**
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/README.md`
- Modify: `imitation_learning/tests/test_artifacts.py`
- Modify: `imitation_learning/tests/test_training_metrics.py`

**Interfaces:**
- Adds: `LoserAugmentationSettings(enabled: bool, recent_dates: int, expert_ratio: float)` nested in `TrainSettings`.
- Consumes: `load_expert_loser_date_info` and `DatasetSplits` loser accounting from Tasks 1 and 3.
- Logs: per-date `loser_aug_date=...` lines and aggregate `data/loser_augmentation_*` WandB metrics.

- [ ] **Step 1: Write failing configuration and reporting tests**

Assert YAML parsing accepts the three nested fields and rejects invalid types/ranges. Assert recent-date selection excludes the latest cached date and fails when fewer than `recent_dates` training dates exist. Assert formatted per-date output contains `cutoff`, `participant_scores`, `episodes`, `score_eligible_episodes`, `after_validation_episodes`, `selected_train_episodes`, and `loser_samples`.

- [ ] **Step 2: Run focused training tests and verify failure**

Run: `pytest imitation_learning/tests/test_artifacts.py imitation_learning/tests/test_training_metrics.py -q`

Expected: FAIL because loser-augmentation settings and reporting are absent.

- [ ] **Step 3: Implement settings and training data flow**

Parse and validate the nested settings. Select the latest requested number of distinct cache dates after removing `max(dataset.shard_dates)`. Load daily loser manifest information only for those dates when enabled, pass eligible episode keys to `build_splits`, and pass `None` when disabled. Preserve all existing expert-validation loading and masks.

- [ ] **Step 4: Implement terminal and WandB reporting**

Join manifest counts with split counts and print the approved per-date fields. Print aggregate selected dates/replays/samples/fraction. Add corresponding scalar counts under `data/` in the existing initial WandB log; do not call artifact upload APIs.

- [ ] **Step 5: Document rebuild and experiment behavior**

Update README run instructions to require one extraction/cache rebuild for this schema, state that later setting changes need training only, and distinguish `loser_augmentation.expert_ratio` from `expert_validation_ratio`.

- [ ] **Step 6: Run focused training tests**

Run: `pytest imitation_learning/tests/test_artifacts.py imitation_learning/tests/test_training_metrics.py -q`

Expected: PASS.

- [ ] **Step 7: Commit training integration**

```bash
git add imitation_learning/training/train.py imitation_learning/cfg/train.yaml imitation_learning/README.md imitation_learning/tests/test_artifacts.py imitation_learning/tests/test_training_metrics.py
git commit -m "feat: configure expert loser augmentation"
```

### Task 5: Integrated Regression Verification

**Files:**
- Verify only; modify a test only if it exposes a regression in the new behavior.

**Interfaces:**
- Consumes all prior task outputs.
- Produces a verified schema-upgrade and training-split implementation.

- [ ] **Step 1: Run all directly affected tests together**

Run: `pytest imitation_learning/tests/test_expert_validation.py imitation_learning/tests/test_extract_alignment.py imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_history_feature_cache.py imitation_learning/tests/test_option_cache.py imitation_learning/tests/test_artifacts.py imitation_learning/tests/test_training_metrics.py -q`

Expected: PASS.

- [ ] **Step 2: Run static validation**

Run: `python -m compileall -q imitation_learning/training`

Expected: exit code 0.

- [ ] **Step 3: Inspect scope and whitespace**

Run: `git diff --check`

Then run: `git status --short`

Expected: no whitespace errors; only intentional project files are modified.

- [ ] **Step 4: Commit any verification-only correction**

If verification required a correction, commit only that correction with its regression test. Otherwise do not create an empty commit.
