# Validation Probability Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional equal-weight probability ensembling to the standalone validation evaluator while preserving exact single-model behavior.

**Architecture:** Configuration supplies a strict ensemble flag and ordered checkpoint list. Each checkpoint independently restores its model architecture, compatibility is checked through cache signatures, and the metric layer averages FP32 legal-action probabilities within each batch before computing CE and Top-K.

**Tech Stack:** Python 3.10+, PyYAML, NumPy, PyTorch, pytest.

## Global Constraints

- Do not modify training, model, feature, cache, or checkpoint formats.
- Single-model mode must retain the existing logits-based metric path.
- Ensemble mode uses equal-weight FP32 softmax probability averaging after legal-action masking.
- All models remain resident on the configured device.
- Validation splits, subgroup names, and one-forward-per-model-per-parent-batch behavior remain unchanged.

---

### Task 1: Ensemble Configuration

**Files:**
- Modify: `imitation_learning/cfg/validation.yaml`
- Modify: `imitation_learning/validation/config.py`
- Test: `imitation_learning/tests/test_validation_config.py`

**Interfaces:**
- Produces: `ValidationSettings(ensemble_enabled: bool, checkpoints: tuple[Path, ...], ...)`.
- Consumes: the existing `${version_name}` interpolation and project-root path resolver.

- [ ] **Step 1: Write failing configuration tests**

Test that ordered checkpoint paths are resolved and interpolated, false requires exactly one checkpoint, true requires at least two, `enabled` must be boolean, paths must be distinct non-empty strings, and the former singular `checkpoint` field is rejected as incomplete configuration.

- [ ] **Step 2: Run the focused test and verify failure**

Run the existing project test interpreter against `imitation_learning/tests/test_validation_config.py -q`.

Expected: FAIL because `ensemble_enabled` and `checkpoints` do not exist.

- [ ] **Step 3: Implement strict ensemble parsing**

Replace `ValidationSettings.checkpoint` with `ensemble_enabled` and `checkpoints`. Parse:

```yaml
ensemble:
  enabled: false
  checkpoints:
    - outputs/${version_name}/checkpoints/epoch-005.pt
```

Validate cardinality, boolean type, non-empty strings, and uniqueness before resolving paths.

- [ ] **Step 4: Run configuration tests**

Expected: all configuration tests pass.

### Task 2: Probability-Averaged Metrics

**Files:**
- Modify: `imitation_learning/validation/metrics.py`
- Test: `imitation_learning/tests/test_validation_metrics.py`

**Interfaces:**
- Produces: `ensemble_probabilities(logits, action_counts) -> Tensor` for one model and an internal multi-model batch forward path.
- Modifies: `evaluate_dataset(..., models: Sequence[Module], ensemble_enabled: bool, ...)`.

- [ ] **Step 1: Write failing probability ensemble tests**

Use two deterministic counting models with deliberately different logit scales. Assert each model runs once per batch, invalid actions receive zero probability, probabilities are averaged rather than logits, CE equals `-mean(log(mean_probability[target]))`, Top-1/3/5 use the averaged probabilities, and one-model mode matches the prior metrics exactly.

- [ ] **Step 2: Run metric tests and verify failure**

Expected: FAIL because multi-model evaluation is unsupported.

- [ ] **Step 3: Implement ensemble prediction and accumulation**

In ensemble mode, run every model on the same batch under autocast, cast logits to FP32, mask invalid actions, softmax, accumulate, divide by model count, and pass `log(clamp_min(torch.finfo(torch.float32).tiny))` into the unchanged metric calculation. Preserve the current direct-logits branch when disabled. Restore every model's prior train/eval state.

- [ ] **Step 4: Run metric tests**

Expected: all metric tests pass.

### Task 3: Multi-Checkpoint Orchestration and Documentation

**Files:**
- Modify: `imitation_learning/validation/evaluate.py`
- Modify: `imitation_learning/README.md`
- Test: `imitation_learning/tests/test_validation_evaluate.py`

**Interfaces:**
- Produces: `validate_compatible_configs(configs: Sequence[ModelConfig]) -> dict`.
- Consumes: ordered `settings.checkpoints`, `_build_model`, `feature_signature`, and ensemble-aware `evaluate_dataset`.

- [ ] **Step 1: Write failing orchestration tests**

Assert compatible models may differ in depth/width, incompatible cache signatures fail with checkpoint indices, checkpoint order is retained, and evaluation calls receive all models plus the configured ensemble flag.

- [ ] **Step 2: Run orchestration tests and verify failure**

Expected: FAIL because compatibility and multiple model loading are absent.

- [ ] **Step 3: Implement multi-checkpoint loading**

Load every checkpoint and config in configuration order. Compare each `feature_signature` to the first, build each model independently, move all models to the device, and print checkpoint path plus recovered architecture. Open the mmap cache with the shared signature and pass the model tuple to every validation call.

- [ ] **Step 4: Update README usage**

Document strict single/ensemble cardinality, probability averaging, differing architecture support, compatibility requirements, and summed GPU memory usage.

- [ ] **Step 5: Run focused and integration verification**

Run all three standalone validation test files together, compile `imitation_learning/validation`, load the real validation YAML, run `git diff --check`, and inspect `git status --short`.

Expected: focused tests pass, compilation exits zero, configuration reports the ordered checkpoint count, and no unrelated source file changes appear.
