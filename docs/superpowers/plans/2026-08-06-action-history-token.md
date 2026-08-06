# Action History Token Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode the acting player's previous three selected actions as one optional encoder token with independently trained basic, structural, and full history representations.

**Architecture:** Cache a superset of the three history representations: decision-level select type/context, selected-option decoder features without numeric indices, and normalized structural fields. The model selects one history representation, pools each historical action, applies one shared per-action MLP, concatenates the three chronological action vectors, and maps them to one encoder token. Training and Kaggle inference use the same rolling three-action history.

**Tech Stack:** Python 3.10, NumPy packed mmap cache, PyTorch, PyYAML, Jupyter notebook JSON.

## Global Constraints

- History modes are exactly `off`, `basic`, `structural`, and `full`.
- The history window is fixed at three actions ordered `[t-3, t-2, t-1]`.
- History embeddings and projections never share trainable parameters with the current decoder.
- `full` excludes option index, tool index, energy index, in-play index, and option relative position.
- The fixed card and attack feature tables may be reused as non-trainable data.
- One `history_action_mlp` is shared across all three time positions.
- The three action vectors are concatenated and passed through `history_sequence_mlp`; no CNN, RNN, or history self-attention is introduced.
- Histories are isolated by replay and player and are cleared at game boundaries.
- Existing extracted JSONL is sufficient; rebuilding the packed feature cache is required.

---

### Task 1: Historical action feature extraction

**Files:**
- Modify: `imitation_learning/model/features.py`
- Create: `imitation_learning/tests/test_action_history_features.py`

**Interfaces:**
- Produces: `HistoryActionFeatures`, `history_action_features(obs, selected, card_count, attack_count, numeric_catalog=None)`.
- `HistoryActionFeatures` contains decision scalars plus selected-option arrays, not model embeddings.

- [ ] **Step 1: Write failing tests for basic, structural, full, combination, and empty actions**

Create small typed observations with `SimpleNamespace` and assert:

```python
history = history_action_features(obs, [0, 2], card_count, attack_count, catalog)
assert history.select_type == int(obs.select.type)
assert history.select_context == int(obs.select.context)
assert history.option_categorical.shape == (2, OPTION_CATEGORICAL_DIM)
assert history.structural.shape == (2, HISTORY_STRUCTURAL_DIM)
assert history.pokemon_dynamic.shape == (2, POKEMON_DYNAMIC_DIM)
assert history.attack_dynamic.shape == (2, ATTACK_DYNAMIC_DIM)
```

Cover normalized `PLAY` source `HAND`, normalized `ATTACK` own/opponent Active relations, and `selected=[]` producing zero option rows while retaining select type/context.

- [ ] **Step 2: Run the focused test and confirm the missing API failure**

Run:

```powershell
python -m pytest imitation_learning/tests/test_action_history_features.py -q
```

Expected: import failure for `HistoryActionFeatures` or `history_action_features`.

- [ ] **Step 3: Add fixed history feature definitions**

Add:

```python
HISTORY_STEPS = 3
HISTORY_STRUCTURAL_DIM = 8

@dataclass(frozen=True)
class HistoryActionFeatures:
    select_type: int
    select_context: int
    option_categorical: np.ndarray
    structural: np.ndarray
    pokemon_dynamic: np.ndarray
    attack_dynamic: np.ndarray
```

Use structural column order:

```text
option_type, source_area, target_area, source_relation,
target_relation, number, count, special_condition
```

- [ ] **Step 4: Implement normalized structural routing and selected-row extraction**

Call `decoder_features(obs, [selected], ...)` once, select only the requested option rows, and build structural fields from normalized OptionType semantics. Validate unique in-range selected indices. Do not copy `decoder.numeric` into the returned object.

- [ ] **Step 5: Run focused tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add imitation_learning/model/features.py imitation_learning/tests/test_action_history_features.py
git commit -m "feat: extract semantic action history features"
```

---

### Task 2: Packed cache history schema and replay-local tracker

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/training/cache_features.py`
- Create: `imitation_learning/tests/test_history_feature_cache.py`

**Interfaces:**
- Extends `FeatureRecord`, `FeatureView`, and `CachedBatch` with fixed decision arrays and variable selected-option history arrays.
- Produces `ActionHistoryTracker.push(record, obs, action_features)` and `ActionHistoryTracker.previous(record) -> tuple[HistoryActionFeatures, ...]` behavior local to one replay/player.

- [ ] **Step 1: Write failing cache round-trip tests**

Write a temporary one-record shard containing three history slots and assert `PackedShard.sample()` and `MmapFeatureDataset.collate()` preserve:

```text
history_select_type       [batch, 3]
history_select_context    [batch, 3]
history_valid             [batch, 3]
history_option_offset     [batch*3 + 1]
history_option_categorical [selected rows, 11]
history_structural         [selected rows, 8]
history_pokemon_dynamic    [selected rows, 46]
history_attack_dynamic     [selected rows, 6]
```

Also test that a new episode and a different player do not inherit history.

- [ ] **Step 2: Run the focused cache test and confirm schema failure**

```powershell
python -m pytest imitation_learning/tests/test_history_feature_cache.py -q
```

Expected: dataclass constructor or missing-section failure.

- [ ] **Step 3: Upgrade packed schema**

Increment `CACHE_SCHEMA_VERSION` and add section dtypes for decision values, valid flags, history option pointers, history categorical/structural values, and history dynamic values. Validate all fixed widths and monotonically increasing pointers in writer and reader.

- [ ] **Step 4: Extend sample views and batch collation**

Map each record's three local option ranges into one batch-level `history_option_offset`, adjusting offsets while concatenating the variable arrays. Empty historical actions retain equal consecutive offsets.

- [ ] **Step 5: Build histories during source-order cache conversion**

In `process_source`, maintain deques keyed within the current episode by player. For each JSONL record:

1. read the previous three actions before preparing the current sample;
2. prepare/write the current sample with left-padded history;
3. append the current real action even when its current label is outside the first 64 enumerated actions, provided its selected indices are valid;
4. clear player deques when the episode changes.

Update `feature_signature()` with the new history layout. Keep the extracted replay schema requirement at version 3.

- [ ] **Step 6: Run cache tests and existing feature tests**

```powershell
python -m pytest imitation_learning/tests/test_action_history_features.py imitation_learning/tests/test_history_feature_cache.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```powershell
git add imitation_learning/training/feature_cache.py imitation_learning/training/cache_features.py imitation_learning/tests/test_history_feature_cache.py
git commit -m "feat: cache three-step action histories"
```

---

### Task 3: Independent history encoders and encoder-token integration

**Files:**
- Modify: `imitation_learning/model/network.py`
- Create: `imitation_learning/tests/test_action_history_network.py`

**Interfaces:**
- Extends `ModelConfig` with `history_encoding`, `history_action_mlp_layers`, and `history_sequence_mlp_layers`.
- Extends `PTCGTransformer.forward()` with all history tensors from `CachedBatch`.
- Produces exactly 26 encoder tokens for `off` and 27 for enabled modes.

- [ ] **Step 1: Write failing model tests for all four modes**

Instantiate tiny models and assert:

```python
assert off_model.encoder_token_count == 26
assert basic_model.encoder_token_count == 27
assert structural_model.encoder_token_count == 27
assert full_model.encoder_token_count == 27
```

Patch or hook `history_action_mlp` to verify it is called with one stacked tensor rather than three separate modules. For `full`, change only the five current option numeric inputs and assert the history token is unchanged.

- [ ] **Step 2: Run focused model tests and confirm config failure**

```powershell
python -m pytest imitation_learning/tests/test_action_history_network.py -q
```

Expected: `ModelConfig` rejects or lacks history fields.

- [ ] **Step 3: Add validated history configuration**

Validation rules:

```text
history_encoding in {off,basic,structural,full}
history_action_mlp_layers >= 0
history_sequence_mlp_layers >= 1 when enabled
```

Only construct modules needed by the selected mode.

- [ ] **Step 4: Implement independent basic and structural option encoders**

Use dedicated `history_*` embedding tables. Pool selected option rows into `batch*3` action slots using `F.embedding_bag(..., mode="sum")`, then add the decision embeddings once per valid action and `history_no_action_embedding` for a valid empty selection.

- [ ] **Step 5: Implement independent full option encoder**

Create history-owned Card ID, Attack ID, categorical, static Card/Attack, Pokémon dynamic, and attack dynamic parameters. Reuse only fixed feature buffers. The function accepts no history numeric tensor and cannot call `option_numeric_projection`.

- [ ] **Step 6: Implement shared action MLP, chronological fusion, and mask**

Reshape pooled actions to `[batch, 3, d_model]`, apply the same action MLP to the full tensor, multiply by `history_valid`, concatenate to `[batch, 3*d_model]`, and apply the sequence MLP. Append the result after the original 26 encoder tokens and append one padding-mask column equal to `~history_valid.any(dim=1)`.

- [ ] **Step 7: Run focused and combined tests**

```powershell
python -m pytest imitation_learning/tests/test_action_history_network.py imitation_learning/tests/test_history_feature_cache.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

```powershell
git add imitation_learning/model/network.py imitation_learning/tests/test_action_history_network.py
git commit -m "feat: add configurable action history encoder token"
```

---

### Task 4: Training configuration and batch plumbing

**Files:**
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/README.md`
- Modify: `imitation_learning/training/export_inference.py` only if checkpoint filtering assumes a fixed config schema.

**Interfaces:**
- `ModelSettings` and checkpoint `ModelConfig` carry the three history settings.
- `_forward_batch()` transfers every history array to the model.

- [ ] **Step 1: Add settings validation tests to the network/config test module**

Assert invalid mode, negative action MLP depth, and zero enabled sequence depth fail with explicit messages.

- [ ] **Step 2: Add YAML and dataclass fields**

Use defaults:

```yaml
history_encoding: off
history_action_mlp_layers: 1
history_sequence_mlp_layers: 2
```

Pass them unchanged into `ModelConfig` so checkpoints remain architecture-self-describing.

- [ ] **Step 3: Pass cached history tensors through `_forward_batch()`**

Use `torch.long` for all categorical/offset/valid inputs and `torch.float32` for dynamic arrays under autocast, matching existing option features.

- [ ] **Step 4: Document modes and cache rebuild requirement**

Document that switching among basic/structural/full reuses the same new cache, while any cache with the previous schema must be rebuilt by `training/cache_features.py`; raw replay extraction is not required.

- [ ] **Step 5: Run configuration and model tests**

```powershell
python -m pytest imitation_learning/tests/test_action_history_network.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```powershell
git add imitation_learning/training/train.py imitation_learning/cfg/train.yaml imitation_learning/README.md imitation_learning/training/export_inference.py
git commit -m "feat: configure action history training"
```

---

### Task 5: Kaggle inference parity and final verification

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Modify: `imitation_learning/tests/test_action_history_network.py` if a reusable online tracker helper is introduced.

**Interfaces:**
- Generated `main.py` reconstructs the history architecture solely from checkpoint config.
- The agent stores its own previous three returned actions and clears them when a new deck/game begins.

- [ ] **Step 1: Add online rolling-history behavior to generated agent code**

Before inference, convert the deque to the same padded tensors consumed during training. After choosing and returning an action, derive its history feature from the current observation and append it. On `obs.select is None`, clear history before returning the deck. Also clear when the observed step decreases if step metadata is available.

- [ ] **Step 2: Mirror all independent history modules and forward arguments**

The notebook's `ModelConfig`, constants, feature extraction, network constructor, and forward call must match project code exactly for all four modes. `full` must not create or pass history numeric features.

- [ ] **Step 3: Validate notebook JSON and generated code syntax**

Run:

```powershell
python -m json.tool imitation_learning/kaggle_submission_imitation_agent.ipynb > $null
```

Then execute or extract the notebook's generated `main.py` cell and run `compile(source, "main.py", "exec")` without loading a model.

- [ ] **Step 4: Run the complete focused suite**

```powershell
python -m pytest imitation_learning/tests/test_action_history_features.py imitation_learning/tests/test_history_feature_cache.py imitation_learning/tests/test_action_history_network.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run static verification**

```powershell
python -m compileall imitation_learning/model imitation_learning/training
git diff --check
git status --short
```

Expected: compilation succeeds, no whitespace errors, and only intentional changes remain.

- [ ] **Step 6: Commit Task 5**

```powershell
git add imitation_learning/kaggle_submission_imitation_agent.ipynb imitation_learning/tests
git commit -m "feat: support action history in Kaggle agent"
```
