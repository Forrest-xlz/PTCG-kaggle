# Final Full-Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `ver_final_f` to a concise full-data training workflow matching `ver_final_0` while preserving the current model and features.

**Architecture:** Replace validation-oriented dataset splitting with one full-training index builder that optionally appends expert-loser samples. Remove train-time validation configuration/evaluation, while retaining all optimizer, logging, checkpoint, and model construction code.

**Tech Stack:** Python, NumPy, PyTorch, PyYAML, unittest/pytest, W&B.

## Global Constraints

- Preserve cache schema `19`, summary dimensions `86/84`, and all current model features.
- Keep expert-loser augmentation identical to `ver_final_0`.
- Keep the standalone validation package and Kaggle notebook unchanged.
- Do not add a `full_training` compatibility switch.

---

### Task 1: Full-training dataset selection

**Files:**
- Modify: `imitation_learning/tests/test_feature_cache.py`
- Modify: `imitation_learning/training/feature_cache.py`

**Interfaces:**
- Produces: `MmapFeatureDataset.build_training_indices(...)` returning all normal samples plus configured loser samples and reporting augmentation counts.

- [ ] Add/restore tests proving no date/replay/deck validation exclusion occurs.
- [ ] Run the focused tests and observe failure against validation splitting.
- [ ] Port the `ver_final_0` full-training selection behavior while preserving schema-19 record layout.
- [ ] Run cache tests and confirm success.

### Task 2: Final training configuration and loop

**Files:**
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/tests/test_training_schedule.py`

**Interfaces:**
- Consumes: `build_training_indices(loser_episode_keys, max_samples, replay_ratio, replay_seed)`.
- Produces: a train-only epoch loop with checkpointing, W&B train metrics, and no validation evaluation.

- [ ] Add source/config tests defining the absence of train-time validation and presence of expert-loser augmentation.
- [ ] Run them and confirm failure.
- [ ] Remove validation-only settings, imports, helpers, split construction, evaluation calls, and W&B namespaces.
- [ ] Keep model construction, optimizer, scheduler, resume, logging, and save behavior unchanged.
- [ ] Run training configuration/schedule tests and confirm success.

### Task 3: Regression and artifact verification

**Files:**
- Verify: `imitation_learning/model/features.py`
- Verify: `imitation_learning/model/network.py`
- Verify: `imitation_learning/training/feature_cache.py`
- Verify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`

**Interfaces:**
- Produces: a clean `ver_final_f` branch ready for cache-19 full-data training.

- [ ] Run feature-cache, setup-summary, known-deck, schedule, and notebook tests.
- [ ] Compile Python sources and validate YAML/notebook syntax.
- [ ] Run `git diff --check`, inspect the diff against `ver_final_0`, and commit.

