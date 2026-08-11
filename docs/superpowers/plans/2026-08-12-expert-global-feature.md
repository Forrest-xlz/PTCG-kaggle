# Expert-Conditioned Global Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append a replay-level expert condition to the global summary, cache it using daily validation-style score rankings, and expose a manual condition switch in the Kaggle notebook.

**Architecture:** Cache construction references the training YAML, loads per-date expert episode keys before multiprocessing, and passes the matching set into each source worker. The model consumes a 74-dimensional global summary, while cache, training, and standalone validation signatures include the expert ratio. Kaggle inference appends a configurable boolean instead of attempting to infer unavailable future ranking information.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, PyYAML, packed mmap cache, nbformat JSON, pytest.

## Global Constraints

- Preserve extracted JSONL and all existing 73 global feature positions.
- Append `is_expert` only at index 73.
- Use `load_expert_date_info` and `train.expert_validation_ratio` without a second ranking implementation.
- Increment cache schema and require recache, but not re-extraction.
- Do not add backward compatibility for old checkpoints.
- Keep encoder/decoder/Transformer behavior otherwise unchanged.

---

### Task 1: 74-Dimensional Feature Contract

**Files:**
- Modify: `imitation_learning/model/features.py`
- Modify: `imitation_learning/model/network.py`
- Modify: `imitation_learning/training/feature_cache.py`
- Test: `imitation_learning/tests/test_expert_global_feature.py`

**Interfaces:**
- Produces: `encoder_features(..., is_expert: bool = False)` and `GLOBAL_SUMMARY_DIM = 74` across feature generation, storage, batching, and network projection.

- [ ] Write failing tests proving the original 73 values are unchanged, index 73 is exactly zero/one, default inference is non-expert, packed records require 74 values, and the model global projection accepts 74 inputs.
- [ ] Run the focused test and confirm dimension/signature failures.
- [ ] Add the keyword-only flag, append it after all existing values, update error text and constants, and increment `CACHE_SCHEMA_VERSION` from 16 to 17.
- [ ] Run feature, cache, and model tests.

### Task 2: Daily Expert Labels During Parallel Cache Build

**Files:**
- Modify: `imitation_learning/cfg/cache.yaml`
- Modify: `imitation_learning/training/cache_features.py`
- Test: `imitation_learning/tests/test_cache_expert_labels.py`

**Interfaces:**
- Extends: `CacheSettings(..., train_config: str)`.
- Produces: `feature_signature(config, expert_ratio)`, worker jobs containing one source date's expert episode keys, and `_prepare_record(..., is_expert)`.

- [ ] Write failing tests for referenced training-config loading, missing paths, daily key routing, expert and non-expert records, ratio in signature, and worker rejection when a source date has no key set.
- [ ] Run the focused tests and confirm the old cache configuration and worker tuple fail.
- [ ] Load the referenced train settings at cache startup, resolve replay archives, call `load_expert_date_info` once for all source dates, add ratio and `global_summary_layout: expert-conditioned-v1` to the signature, and pass per-date immutable keys in each job.
- [ ] Make `_prepare_record` append the label based on `stable_episode_key(record["episode_id"])`; preserve history and skip behavior.
- [ ] Run cache tests including multiprocessing-compatible job serialization.

### Task 3: Training and Standalone Validation Signature Enforcement

**Files:**
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/validation/evaluate.py`
- Test: `imitation_learning/tests/test_expert_signature.py`

**Interfaces:**
- Changes: `training.train.feature_signature(config, expert_ratio)`.
- Changes: standalone validation augments the model compatibility signature with `settings.expert_validation_ratio` before opening `MmapFeatureDataset`.

- [ ] Write failing tests proving train/checkpoint/cache signatures carry the configured ratio, standalone validation expects the same value, changing only the ratio rejects the cache, and model-to-model ensemble compatibility remains architecture-based.
- [ ] Run focused signature tests and confirm missing ratio failures.
- [ ] Pass `train_cfg.expert_validation_ratio` to every training signature call, and add the ratio/layout fields to the standalone dataset signature after model compatibility validation.
- [ ] Run training, feature-cache, and standalone-validation regressions.

### Task 4: Kaggle Notebook Expert Switch

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Modify: `imitation_learning/README.md`
- Test: `imitation_learning/tests/test_artifacts.py`

**Interfaces:**
- Produces: editable `IS_EXPERT = True`, notebook `GLOBAL_SUMMARY_DIM = 74`, and live `global_summary(..., is_expert)` output with the condition at index 73.

- [ ] Write failing notebook artifact tests for the parameter, boolean assertion, 74-dimensional constant, flag propagation into `encoder_features`, and final appended value.
- [ ] Run the artifact test and confirm the old notebook fails.
- [ ] Patch notebook JSON cells without changing submission packaging, checkpoint auto-architecture loading, ensemble behavior, deck configuration, or engine integration.
- [ ] Document the condition semantics, ratio coupling, recache requirement, old-checkpoint incompatibility, and `IS_EXPERT=False` ablation.
- [ ] Run all focused and relevant regression tests, validate notebook JSON/nbformat, compile Python, load real YAML, run `git diff --check`, and inspect modified-file scope.
