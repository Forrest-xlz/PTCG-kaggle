# Isolated Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone fine-tuning workflow that preserves the current validation split and trains on a replay-sampled generic subset plus duplicated expert winning samples from one exact deck.

**Architecture:** `cfg/finetune.yaml` references the existing training YAML for data, validation, optimizer defaults, and runtime settings. A new isolated `finetune` package validates its own configuration, builds full and replay-sampled versions of the unchanged split, selects expert-deck winners from the full post-validation train set, restores model and optimizer state from a complete epoch checkpoint, creates a fresh scheduler, and reuses existing batch, metric, validation, and checkpoint conventions.

**Tech Stack:** Python 3.10+, PyYAML, NumPy, PyTorch, mmap feature cache, pytest.

## Global Constraints

- Do not change `training/train.py`, model architecture, extraction, or cache schemas.
- Validation indices and subgroup masks must match normal training for the same referenced configuration.
- Generic fine-tune sampling is replay-level and deterministic.
- Expert cutoffs are computed independently per date with existing expert-validation quantile behavior.
- Expert-deck additions require exact deck, expert replay, and winning acting player.
- Overlap is intentionally duplicated.
- Restore model, optimizer, and compatible scaler; reset scheduler, EMA, history, and global step.

---

### Task 1: Fine-Tune Configuration

**Files:**
- Create: `imitation_learning/finetune/__init__.py`
- Create: `imitation_learning/finetune/finetune.py`
- Create: `imitation_learning/cfg/finetune.yaml`
- Test: `imitation_learning/tests/test_finetune_config.py`

**Interfaces:**
- Produces: `FineTuneSettings(version_name, train_config, checkpoint, output, epochs, in_distribution_ratio, in_distribution_seed, expert_ratio, deck, learning_rate, wandb)` and `load_finetune_settings(path)`.

- [ ] Write failing tests that load the proposed YAML, interpolate `${version_name}`, resolve inherited learning rate, and reject missing paths, ratios outside `(0, 1]`, non-positive epochs, invalid optional learning rate, and decks that are not exactly 60 non-negative integers.
- [ ] Run `pytest imitation_learning/tests/test_finetune_config.py -q` and confirm collection fails because the new package is absent.
- [ ] Implement strict nested-YAML parsing. Load the referenced config with `training.train.load_settings`, use its training learning rate when the optional fine-tune override is null, and reject unknown fields.
- [ ] Add the commented baseline YAML with one exact 60-card deck and disabled W&B defaults.
- [ ] Run the focused configuration tests.

### Task 2: Fine-Tune Index Construction

**Files:**
- Modify: `imitation_learning/finetune/finetune.py`
- Test: `imitation_learning/tests/test_finetune_data.py`

**Interfaces:**
- Produces: `FineTuneData(indices, generic_indices, expert_deck_indices, overlap_samples)` and `build_finetune_indices(dataset, full_train_indices, generic_train_indices, expert_episode_keys, deck_key)`.

- [ ] Write fake-shard tests proving the function retains the generic order, filters expert additions by date, exact acting-player `deck_key`, and `PLAYER_RESULT_WIN`, excludes samples outside `full_train_indices`, duplicates overlap after concatenation, reports overlap count, and rejects an empty generic or expert-deck component.
- [ ] Run the focused test and confirm the new API is missing.
- [ ] Walk shard ranges with `searchsorted`; read only cached `episode_key`, `deck_key`, and `player_result`; collect qualifying global indices and concatenate generic plus expert-deck arrays without deduplication.
- [ ] Add `epoch_finetune_indices(data, seed)` that copies and globally shuffles the combined indices, with deterministic same-seed tests.
- [ ] Run fine-tune data tests.

### Task 3: Checkpoint Initialization and Fresh Schedule

**Files:**
- Modify: `imitation_learning/finetune/finetune.py`
- Test: `imitation_learning/tests/test_finetune_checkpoint.py`

**Interfaces:**
- Produces: `load_finetune_checkpoint(path) -> dict`, `restore_finetune_state(checkpoint, model, optimizer, precision)`, and `create_finetune_scheduler(optimizer, epochs, sample_count, batch_size, warmup_steps)`.

- [ ] Write failing tests requiring an `epoch-NNN.pt` file with `model`, `optimizer`, `scaler`, and `config`; reject inference-only or malformed checkpoints; prove model and optimizer states restore while the returned scheduler starts at a new schedule step zero.
- [ ] Run the focused checkpoint tests and confirm failure for missing APIs.
- [ ] Load trusted local checkpoints with `weights_only=False` and the older-PyTorch fallback. Construct `ModelConfig(**checkpoint["config"])`; load model, optimizer, and scaler without loading checkpoint scheduler, EMA, history, global step, or RNG state.
- [ ] After loading the optimizer state, set every parameter group's learning rate to the resolved fine-tune target and build the existing warmup-cosine scheduler using `ceil(sample_count / batch_size) * epochs`.
- [ ] Run checkpoint tests.

### Task 4: Standalone Training and Validation Entry Point

**Files:**
- Modify: `imitation_learning/finetune/finetune.py`
- Modify: `imitation_learning/README.md`
- Test: `imitation_learning/tests/test_finetune_integration.py`

**Interfaces:**
- Consumes: Tasks 1-3 and existing `train_batch`, `evaluate_dataset`, `PolicyMetrics`, `ExponentialMovingAverage`, `top_deck_subgroup_masks`, feature signature, isolation loader, and expert manifest loader.
- Produces: runnable `python imitation_learning/finetune/finetune.py`.

- [ ] Write source/integration tests proving the full split uses `train_replay_ratio=1.0`, the generic split uses `in_distribution_ratio`, both use identical validation inputs, validation arrays are checked for equality, expert-deck filtering uses the full train indices, checkpoint model config is authoritative, scheduler is fresh, epoch shuffle uses `seed + epoch_index`, and no production training/model/cache file is modified.
- [ ] Run the integration tests and confirm the entry point is incomplete.
- [ ] Bootstrap paths from `cfg/finetune.yaml`, load the referenced training settings, cache, isolation selections, daily validation experts, optional loser augmentation, full split, generic split, and independent daily fine-tune expert sets. Assert all validation indices and masks match between the two split builds.
- [ ] Construct the checkpoint architecture and static feature tables, restore model/optimizer/scaler, initialize fresh EMA and history, and train with existing mixed-precision policy loss and logging cadence.
- [ ] Reuse all current isolation/latest/in-distribution validation groups and Top-1/3/5 metrics. Define matching W&B namespaces, log fine-tune data counts and daily cutoffs, and do not upload checkpoint artifacts.
- [ ] Save complete fine-tune step and epoch checkpoints containing the same core keys (`model`, `optimizer`, `scheduler`, `scaler`, `global_step`, `epoch`, `config`, `ema`, `history`) plus resolved fine-tune experiment metadata.
- [ ] Document configuration, dataset composition, overlap weighting, checkpoint requirements, fresh scheduler behavior, execution command, and lack of extract/cache rebuild.
- [ ] Run all new focused tests, existing training/validation regressions, Python compilation, real YAML loading, and `git diff --check`; inspect that only the new package, YAML, README, and tests changed.
