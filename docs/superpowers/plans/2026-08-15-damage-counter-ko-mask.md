# Damage-Counter KO Action Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable training/validation and Kaggle-inference masks that prevent damage-counter placement on Bench Pokemon at zero or negative HP whenever a positive-HP target remains.

**Architecture:** A pure feature-layer helper computes one eligibility bit per enumerated action from the observation and raw action membership. Schema-18 packed caches persist those bits; training and validation optionally combine them with the existing action-count mask and exclude rows whose recorded label is ineligible. Kaggle inference computes the same bits directly and optionally masks ensemble probabilities before argmax.

**Tech Stack:** Python 3.10, NumPy, PyTorch, mmap packed caches, PyYAML, Jupyter notebook JSON.

## Global Constraints

- Apply the rule only to API select context `DAMAGE_COUNTER_ANY` (numeric value 14).
- An action is ineligible when any selected target Pokemon has `hp <= 0`.
- If every enumerated action is ineligible, mark every action eligible as a legal fallback.
- Preserve raw options, action ordering, policy target indices, decoder features, and action history.
- Training and validation share `train.damage_counter_ko_mask`; Kaggle inference uses `DAMAGE_COUNTER_KO_MASK`.
- Old extracted JSONL remains usable; packed feature caches must be rebuilt once.
- Do not add checkpoint architecture parameters or change the network.

---

### Task 1: Pure damage-counter action eligibility

**Files:**
- Modify: `imitation_learning/model/features.py`
- Create: `imitation_learning/tests/test_damage_counter_mask.py`

**Interfaces:**
- Produces: `damage_counter_action_eligibility(obs: Any, actions: list[list[int]]) -> np.ndarray` with shape `(len(actions),)` and dtype `np.bool_`.
- Non-target contexts return all `True`.

- [ ] **Step 1: Write failing rule tests**

Build `SimpleNamespace` observations containing Bench Pokemon and CARD options.
Assert the exact behavior:

```python
actions = [[0], [1]]
mask = damage_counter_action_eligibility(
    observation(context=14, bench_hps=[0, 40]), actions
)
np.testing.assert_array_equal(mask, [False, True])

fallback = damage_counter_action_eligibility(
    observation(context=14, bench_hps=[0, -10]), actions
)
np.testing.assert_array_equal(fallback, [True, True])

combined = damage_counter_action_eligibility(
    observation(context=14, bench_hps=[0, 40]), [[0, 1], [1]]
)
np.testing.assert_array_equal(combined, [False, True])

unchanged = damage_counter_action_eligibility(
    observation(context=13, bench_hps=[0, 40]), actions
)
np.testing.assert_array_equal(unchanged, [True, True])
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\global-chess-2025\python.exe' -m pytest imitation_learning/tests/test_damage_counter_mask.py -q
```

Expected: import failure because `damage_counter_action_eligibility` does not exist.

- [ ] **Step 3: Implement the minimal pure helper**

Resolve only `option.area == AreaType.BENCH` targets using `option.playerIndex`
and `option.index`. Treat missing/unresolvable Pokemon as eligible rather than
inventing a restriction. Compute action eligibility with `all(option_eligible[i]
for i in action)`, then apply the all-ineligible fallback.

- [ ] **Step 4: Run the rule tests and verify GREEN**

Run the Task 1 command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add imitation_learning/model/features.py
git add -f imitation_learning/tests/test_damage_counter_mask.py
git commit -m "feat: classify damage counter actions targeting ko pokemon"
```

### Task 2: Persist eligibility and apply the training switch

**Files:**
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/training/feature_cache.py`
- Create: `imitation_learning/training/action_mask.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/validation/config.py`
- Modify: `imitation_learning/validation/metrics.py`
- Modify: `imitation_learning/validation/evaluate.py`
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/tests/test_feature_cache.py`
- Modify: `imitation_learning/tests/test_training_metrics.py`
- Create: `imitation_learning/tests/test_damage_counter_cache.py`
- Create: `imitation_learning/tests/test_damage_counter_training.py`

**Interfaces:**
- `FeatureRecord.action_eligible: list[int]` contains one binary value per action.
- `FeatureView.action_eligible` has shape `(action_count,)`.
- `CachedBatch.action_eligible` has shape `(batch, MAX_ACTIONS)`; padded columns are `False`.
- `policy_metrics(..., action_eligible: torch.Tensor | None = None)` uses the mask only when non-`None`.
- `prepare_policy_batch(logits, targets, action_counts, action_eligible=None)` returns masked logits, retained targets, and the retained-row mask without importing the training runner.
- `TrainSettings.damage_counter_ko_mask: bool` and `ValidationSettings.damage_counter_ko_mask: bool` are required Boolean settings.

- [ ] **Step 1: Write failing cache round-trip tests**

Extend cache fixtures with `action_eligible=[0, 1, ...]`. Assert writer/shard
round-trip equality, binary validation, exact action alignment, and padded batch
shape. Add an integration test that `_prepare_record` stores the pure helper's
result without changing `target`, `action_count`, or history.

- [ ] **Step 2: Write failing metric/config tests**

Test the dependency-light `training.action_mask` helper directly:

```python
masked, retained_targets, retained_rows = prepare_policy_batch(
    logits=torch.tensor([[9.0, 1.0], [9.0, 1.0]]),
    targets=torch.tensor([1, 0]),
    action_counts=torch.tensor([2, 2]),
    action_eligible=torch.tensor([[False, True], [False, True]]),
)
assert retained_rows.tolist() == [True, False]
assert retained_targets.tolist() == [1]
assert masked.argmax(dim=1).tolist() == [1]
```

The first row remains and has its bad high-logit action masked; the second row
is excluded because its label is ineligible. Also assert `action_eligible=None`
keeps both original rows and that non-Boolean YAML values are rejected.

- [ ] **Step 3: Run Task 2 tests and verify RED**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\global-chess-2025\python.exe' -m pytest imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_damage_counter_cache.py imitation_learning/tests/test_damage_counter_training.py -q
```

Expected: failures for missing cache fields and metric/config parameters.

- [ ] **Step 4: Add schema-18 packed storage**

Raise `CACHE_SCHEMA_VERSION` to 18. Add a uint8 `action_eligible` section,
validate one binary value per action, persist it after `action_count`, expose it
through views, and collate to a zero-padded `(batch, MAX_ACTIONS)` array. Update
all tracked cache fixtures.

- [ ] **Step 5: Produce cache eligibility**

Call `damage_counter_action_eligibility(obs, actions)` in `_prepare_record` and
store `.astype(np.uint8).tolist()`. Change every cache/validation feature
signature action-mask identifier to `damage-counter-ko-action-mask-v1` so
schema-17 caches fail early. Do not filter any records or alter history.

- [ ] **Step 6: Apply runtime policy masking**

Add `damage_counter_ko_mask: bool` to train settings and YAML with default
configuration:

```yaml
train:
  damage_counter_ko_mask: true  # Ignore wasteful HP<=0 damage-counter targets.
```

Implement the mask combination in the dependency-light
`training/action_mask.py`; both train and standalone validation import it.
When enabled, move `batch.action_eligible` to the device, combine it with the
action-count mask, and select only rows whose target is eligible before CE and
Top-1/3/5. When no rows survive, return differentiable zero loss and zero-sample
metrics. When disabled, pass `None` and preserve original behavior. Thread the
same setting through periodic validation and standalone validation, which reads
the value from the referenced train YAML.

- [ ] **Step 7: Run Task 2 tests and verify GREEN**

Run the Task 2 command plus:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\global-chess-2025\python.exe' -m pytest imitation_learning/tests/test_revealed_hand_cache.py imitation_learning/tests/test_history_feature_cache.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 2**

```powershell
git add imitation_learning/training/cache_features.py imitation_learning/training/feature_cache.py imitation_learning/training/action_mask.py imitation_learning/training/train.py imitation_learning/validation/config.py imitation_learning/validation/metrics.py imitation_learning/validation/evaluate.py imitation_learning/cfg/train.yaml imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_training_metrics.py
git add -f imitation_learning/tests/test_damage_counter_cache.py imitation_learning/tests/test_damage_counter_training.py
git commit -m "feat: add configurable training mask for ko counter targets"
```

### Task 3: Kaggle inference parity and documentation

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Modify: `imitation_learning/README.md`
- Modify: `imitation_learning/tests/test_submission_notebook.py`

**Interfaces:**
- Notebook top cell exposes `DAMAGE_COUNTER_KO_MASK = True`.
- Notebook `damage_counter_action_eligibility(obs, actions)` matches project output.
- Masking occurs after probability averaging and before argmax, so ensembles obey one common action policy.

- [ ] **Step 1: Write failing notebook parity tests**

Execute the generated notebook namespace without running the live agent. Compare
project and notebook eligibility arrays for positive/zero/negative HP, all-KO
fallback, and non-target context. Assert the top configuration cell contains a
Boolean `DAMAGE_COUNTER_KO_MASK` parameter.

- [ ] **Step 2: Run notebook tests and verify RED**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\global-chess-2025\python.exe' -m pytest imitation_learning/tests/test_submission_notebook.py -q
```

Expected: failure because the notebook helper and parameter are absent.

- [ ] **Step 3: Implement notebook inference masking**

Mirror the pure helper in generated `main.py`. Keep all enumerated actions and
decoder inputs. After ensemble probabilities are averaged, set ineligible
action scores to `-1.0` when the switch is enabled, then call argmax. The
all-ineligible fallback guarantees at least one retained score.

- [ ] **Step 4: Document operation and rebuild requirements**

Update README with both switches, target-label exclusion semantics, the all-KO
fallback, schema 18, and the instruction to rerun only
`training/cache_features.py` once.

- [ ] **Step 5: Run final verification**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\global-chess-2025\python.exe' -m pytest imitation_learning/tests/test_damage_counter_mask.py imitation_learning/tests/test_damage_counter_cache.py imitation_learning/tests/test_damage_counter_training.py imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_submission_notebook.py imitation_learning/tests/test_action_history_network.py imitation_learning/tests/test_revealed_hand_features.py imitation_learning/tests/test_revealed_hand_cache.py imitation_learning/tests/test_revealed_hand_model.py -q
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m compileall -q imitation_learning/model imitation_learning/training imitation_learning/validation
git diff --check
```

Expected: targeted tests pass, compileall exits zero, and diff check prints no errors.

- [ ] **Step 6: Commit Task 3**

```powershell
git add imitation_learning/kaggle_submission_imitation_agent.ipynb imitation_learning/README.md
git add -f imitation_learning/tests/test_submission_notebook.py
git commit -m "feat: mask ko counter targets in kaggle inference"
```
