# Multitask Auxiliary Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional next-decision, opponent Exact Deck, and final-own-prize auxiliary supervision to the existing BC trainer with one shared forward/backward pass.

**Architecture:** Future labels are generated at replay extraction, converted to compact IDs/masks in feature caching, and consumed by small heads attached to the selected decoded action or encoder state representation. The original policy path remains authoritative and auxiliary losses are summed before one backward call.

**Tech Stack:** Python 3.10+, PyTorch, NumPy mmap cache, PyYAML, pytest.

## Global Constraints

- Preserve the user's uncommitted `imitation_learning/eda/opponent_deck_classification.ipynb` changes.
- Three auxiliary tasks are independently enabled and weighted.
- No `OTHER` opponent Deck class; unmatched labels are masked.
- Final-own-prize labels are integer classes 0 through 6.
- Auxiliary-off configuration preserves baseline policy behavior.
- One total loss, one backward, one optimizer step per batch.

---

### Task 1: Extract replay-level future labels

**Files:**
- Modify: `imitation_learning/training/extract.py`
- Test: `imitation_learning/tests/test_multitask_extract.py`

**Interfaces:**
- Produces JSONL fields `next_select_type`, `next_select_context`, `next_decision_valid`, `opponent_deck_id`, and `final_own_prize_count`.

- [ ] Write tests with interleaved player decisions and terminal observations.
- [ ] Verify tests fail because fields are absent.
- [ ] Refactor per-player record generation to materialize records, annotate the next same-player decision, stable opponent Exact Deck ID, and own terminal prize length.
- [ ] Increment extract schema and tighten cache-hit metadata validation.
- [ ] Run extraction tests.

### Task 2: Persist compact labels in feature cache

**Files:**
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/cfg/cache.yaml`
- Test: `imitation_learning/tests/test_multitask_cache.py`

**Interfaces:**
- Consumes extracted labels and opponent-class CSV.
- Produces uint8 labels/masks in `FeatureRecord`, packed shards, `FeatureView`, and `CachedBatch`.

- [ ] Write failing CSV mapping, unmatched-mask, and cache round-trip tests.
- [ ] Add cache configuration for opponent class CSV and load stable Deck-ID-to-class mappings.
- [ ] Add compact label sections and bump cache schema/signature.
- [ ] Run cache tests and existing feature-cache tests.

### Task 3: Add auxiliary model outputs and CLS/global routing

**Files:**
- Modify: `imitation_learning/model/network.py`
- Test: `imitation_learning/tests/test_multitask_model.py`

**Interfaces:**
- Adds auxiliary architecture fields to `ModelConfig`.
- Produces a structured output containing policy logits and optional auxiliary logits while retaining a policy-only inference path.

- [ ] Write failing tests for auxiliary-off, CLS insertion/mask, global selection, selected decoded-action gathering, and output shapes.
- [ ] Add optional learned CLS and four Linear heads.
- [ ] Expose the post-decoder action tensor, gather by BC target only when next-decision training requests it, and preserve policy-only inference behavior.
- [ ] Run model tests.

### Task 4: Integrate configuration, masked losses, metrics, and checkpoints

**Files:**
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/training/train.py`
- Test: `imitation_learning/tests/test_multitask_training.py`

**Interfaces:**
- Consumes the three enabled/weight settings and cached labels/masks.
- Produces one total loss, auxiliary metrics per training/validation namespace, and checkpoint-compatible model configuration.

- [ ] Write failing configuration, masked CE, combined-loss, and metric tests.
- [ ] Parse and validate state representation, weights, task switches, and class CSV consistency.
- [ ] Compute policy plus enabled masked losses inside one autocast block and pass only total loss to the existing single `backward_step` call.
- [ ] Extend EMA/logging/validation aggregation without changing policy metrics.
- [ ] Run training and checkpoint tests.

### Task 5: Regression and operational verification

**Files:**
- Verify modified production and test files only.

**Interfaces:**
- Confirms end-to-end labels, cache, model, and trainer compatibility.

- [ ] Run all new tests plus extraction, cache, model, schedule, precision, metrics, and checkpoint regressions available on this branch.
- [ ] Compile modified Python sources and load both YAML files.
- [ ] Check feature signatures reject old caches and ModelConfig round-trips through checkpoints.
- [ ] Run `git diff --check`, verify the EDA notebook remains uncommitted and untouched by this feature, and commit implementation files only.
