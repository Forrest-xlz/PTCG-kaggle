# Training Sampling Weights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic additive oversampling of winning expert, exact-deck, and expert/exact-deck training samples without changing validation splits or cached features.

**Architecture:** Parse weights into indexed deck rules, load threshold-based expert episode keys from manifests, and build compact rule-specific global-index arrays aligned to the final sorted training split. At every epoch, materialize integer copies plus deterministic fractional samples, concatenate with the full base split, globally shuffle once, and feed existing mmap batches without a second shuffle.

**Tech Stack:** Python 3.10+, PyYAML, NumPy, PyTorch, mmap feature cache, pytest.

## Global Constraints

- Do not change extraction or cache schemas.
- Every base training sample remains once; only winning samples can be added.
- Expert, deck, and expert/deck rules add independently when they overlap.
- `deckN` refers to the one-based order of `train.top_decks` and supports every configured N.
- Fractional sampling changes by epoch but is deterministic from `train.seed + epoch_index`.
- Validation construction and all validation metrics remain unchanged.

---

### Task 1: Configuration Contract

**Files:**
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/training/train.py`
- Test: `imitation_learning/tests/test_training_sampling_config.py`

**Interfaces:**
- Produces: `SamplingSettings(expert_score_threshold: float, expert_weight: float, deck_weights: dict[int, float], expert_deck_weights: dict[int, float])` nested in `TrainSettings`.

- [ ] Write failing tests for the approved YAML, concise comments, weight `>= 1`, finite threshold/weights, exact `deckN` syntax, arbitrary valid indices such as deck4/deck5, rejection beyond `len(top_decks)`, and omitted deck defaults.
- [ ] Run the focused config test and confirm failure because `SamplingSettings` is absent.
- [ ] Parse `train.sampling` before constructing `TrainSettings`, normalize deck names to one-based integer keys, and validate them after `top_decks` is available.
- [ ] Add the commented baseline configuration with every effective weight set to `1.0`.
- [ ] Run the config tests and retain existing `load_settings` behavior.

### Task 2: Threshold Expert Episode Sets

**Files:**
- Modify: `imitation_learning/training/expert_validation.py`
- Test: `imitation_learning/tests/test_expert_validation.py`

**Interfaces:**
- Produces: `ExpertThresholdDateInfo` and `load_expert_threshold_date_info(replay_root, required_dates, threshold)`.

- [ ] Write a failing ZIP-manifest test proving `high_score = sum_score - min_score`, inclusive threshold comparison, date isolation, and finite-threshold validation.
- [ ] Run the focused expert test and confirm the new API is missing.
- [ ] Reuse `_archive_by_date` and `_read_manifest_episodes`; return per-date counts and episode keys whose higher score reaches the absolute threshold.
- [ ] Run expert-validation tests.

### Task 3: Compact Sampling Plan and Epoch Indices

**Files:**
- Create: `imitation_learning/training/sampling.py`
- Test: `imitation_learning/tests/test_training_sampling.py`

**Interfaces:**
- Produces: `SamplingRule`, `TrainingSamplingPlan`, `build_sampling_plan(dataset, train_indices, expert_episode_keys, top_deck_keys, settings)`, `weighted_epoch_size(plan)`, and `build_epoch_indices(plan, seed)`.

- [ ] Write failing tests using small fake shards for winner-only rule membership, losing-sample base-only behavior, deck4 lookup, independent overlap addition, weights 1/1.2/2/3, exact fractional floor counts, deterministic same-epoch output, different fractional samples across epochs, global permutation integrity, and empty active-rule failure.
- [ ] Run focused sampling tests and confirm the module is absent.
- [ ] Require sorted one-dimensional training indices. Walk shard ranges with `searchsorted`, gather only `episode_key`, `deck_key`, and `player_result`, and store global indices only for active rules.
- [ ] Generate integer copies and no-replacement fractional subsets per rule, concatenate with base indices, shuffle once in place, and return the weighted array using the base index dtype.
- [ ] Run sampling tests.

### Task 4: Training Loop, Schedule, and Logging

**Files:**
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/README.md`
- Test: `imitation_learning/tests/test_training_sampling_integration.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: weighted `samples_per_epoch`, startup/W&B data metrics, and one globally shuffled weighted index array per epoch.

- [ ] Write failing source/integration tests that split construction precedes sampling-plan construction, loser augmentation indices remain only in the base, scheduler length uses weighted/truncated count, epoch batches use `shuffle=False`, resume epoch derives the same seed, and validation inputs remain untouched.
- [ ] Load threshold expert data only when expert or expert/deck weight exceeds one. Build the plan after `dataset.build_splits`, print per-date threshold counts and per-rule eligible/added counts, and log equivalent `data/sampling_*` values.
- [ ] Replace base `len(splits.train)` scheduling with `weighted_epoch_size(plan)`, truncated by `max_samples`. At epoch start call `build_epoch_indices(plan, train.seed + epoch_index)` and pass it to `dataset.iter_batches(..., shuffle=False, max_samples=max_samples)`.
- [ ] Document additive overlap semantics, winner-only additions, deck numbering, schedule effects, and that no extract/cache rebuild is required.
- [ ] Run all new focused tests together, compile changed packages, load real `cfg/train.yaml`, run `git diff --check`, and inspect scope.
