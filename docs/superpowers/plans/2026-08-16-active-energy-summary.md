# Active Energy Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add physical and effective Active Energy totals to both player summary tokens.

**Architecture:** Extend the existing public setup-summary helper by two normalized scalars, then propagate the resulting dimensions through packed caches, the model, and the Kaggle notebook. Keep extraction unchanged and invalidate old feature caches explicitly through a schema/signature bump.

**Tech Stack:** Python, NumPy, PyTorch, unittest/pytest, Jupyter JSON.

## Global Constraints

- Own summary records own Active Energy; opponent summary records opponent Active Energy.
- Physical Energy uses `energyCards / 32`; effective Energy uses `energies / 64`.
- Missing Active Pokemon produces zeros.
- Do not modify unrelated model architecture or replay extraction.

---

### Task 1: Active Energy feature behavior

**Files:**
- Modify: `imitation_learning/tests/test_player_setup_summary.py`
- Modify: `imitation_learning/model/features.py`

**Interfaces:**
- Consumes: `_public_setup_summary(player, deck, catalog) -> list[float]`
- Produces: a 13-value public setup summary whose final two values are physical and effective Active Energy.

- [ ] Add tests asserting distinct physical/effective Active Energy values and zero values without Active Pokemon.
- [ ] Run the tests and confirm they fail because only 11 public values exist.
- [ ] Append the two normalized values in `_public_setup_summary`.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Cache and model integration

**Files:**
- Modify: `imitation_learning/model/features.py`
- Modify: `imitation_learning/model/network.py`
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/validation/evaluate.py`
- Modify: `imitation_learning/tests/test_setup_summary_integration.py`

**Interfaces:**
- Produces: own summary dimension `86`, opponent summary dimension `84`, cache schema `19`, and feature signature `numeric-summary-27-known-deck-setup-v5`.

- [ ] Update integration tests with the new dimensions/schema/signature and confirm failure.
- [ ] Update production constants and signatures.
- [ ] Run cache/network integration tests and confirm they pass.

### Task 3: Kaggle inference parity and final verification

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Modify: `imitation_learning/tests/test_submission_setup_summary.py`

**Interfaces:**
- Produces: notebook feature extraction identical to `model/features.py`.

- [ ] Update notebook assertions first and confirm failure.
- [ ] Mirror the two features, dimensions, schema, and signature in the notebook.
- [ ] Run notebook tests, JSON validation, Python compilation, and `git diff --check`.
- [ ] Commit the implementation.

