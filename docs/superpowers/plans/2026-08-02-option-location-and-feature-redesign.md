# Option Location and Auxiliary Feature Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give encoder tokens and decoder options a shared 20-position coordinate system and replace the decoder's old 16 numeric fields with the approved 28-dimensional scalar/one-hot auxiliary representation.

**Architecture:** Cache compact IDs for candidate/target encoder locations and low-cardinality option states. During forward, add the same learned location rows to encoder and option embeddings, expand categorical states to one-hot, concatenate them with three scalars, and project the resulting 28 values to `d_model`. Preserve action-combination summation, cross-attention, static card/attack projections, and policy training.

**Tech Stack:** Python, NumPy, PyTorch, packed mmap cache, Jupyter notebook JSON.

## Global Constraints

- Keep exactly 20 real encoder locations in the current fixed token order.
- Add candidate and target location embeddings directly, without role projections.
- Use a fixed zero padding row only for non-spatial options; do not create a learned no-location token.
- Map hidden zones without a dedicated token to the corresponding player-summary token.
- Remove raw index, player flag, tool/energy indices, in-play fields, option order, linked-entity flag, and card-type scalar from model input.
- Keep normalized attack damage; do not add another attack-energy vector.
- Keep replay JSONL extraction unchanged; require feature-cache rebuild.
- Keep training and Kaggle inference feature contracts identical.

---

### Task 1: Define and test the compact option contract

**Files:**
- Create: `imitation_learning/tests/test_option_location_features.py`
- Modify: `imitation_learning/model/features.py`

**Interfaces:**
- Produces: `OPTION_CATEGORICAL_DIM = 11`, `OPTION_NUMERIC_DIM = 3`, `OPTION_AUXILIARY_DIM = 28`, `ENCODER_TOKENS = 20`, and `NO_ENCODER_LOCATION = 20`.
- Produces: option categorical rows ordered as `(type, context, candidate_id, target_id, attack_id, candidate_location, target_location, area_state, special_condition_state, super_effective_state, resisted_state)`.
- Produces: option numeric rows ordered as `(number, count, attack_damage)`.

- [ ] **Step 1: Add failing tests for state encodings and location resolution**

```python
def test_non_spatial_option_uses_zero_location_row():
    features = decoder_features(make_yes_no_observation(), [[0]], 1300, 500)
    assert features.categorical[0, 5:7].tolist() == [NO_ENCODER_LOCATION] * 2


def test_attach_points_from_hand_to_exact_own_bench_slot():
    features = decoder_features(make_attach_observation(target_slot=3), [[0]], 1300, 500)
    assert features.categorical[0, 5:7].tolist() == [16, 3]


def test_special_condition_distinguishes_missing_from_poison():
    missing = decoder_features(make_end_observation(), [[0]], 1300, 500)
    poison = decoder_features(make_poison_observation(), [[0]], 1300, 500)
    assert missing.categorical[0, 8] == 0
    assert poison.categorical[0, 8] == 1
```

- [ ] **Step 2: Run the focused tests and verify the old contract fails**

Run: `pytest imitation_learning/tests/test_option_location_features.py -q`

Expected: failures showing the old categorical width is 5 and the old numeric width is 16.

- [ ] **Step 3: Add canonical encoder location constants and resolvers**

```python
OWN_BENCH_START = 0
OPPONENT_BENCH_START = 5
OWN_ACTIVE_LOCATION = 10
OPPONENT_ACTIVE_LOCATION = 11
OWN_SUMMARY_LOCATION = 12
OPPONENT_SUMMARY_LOCATION = 13
OWN_DISCARD_LOCATION = 14
OPPONENT_DISCARD_LOCATION = 15
OWN_HAND_LOCATION = 16
OWN_DECK_LOCATION = 17
STADIUM_LOCATION = 18
GLOBAL_SUMMARY_LOCATION = 19
NO_ENCODER_LOCATION = 20
```

Implement a relative-player-aware resolver for ACTIVE, BENCH, HAND, DISCARD,
DECK, STADIUM, PRIZE, PLAYER, LOOKING, and other areas. Own hidden zones fall
back to own summary; opponent hidden zones fall back to opponent summary.

- [ ] **Step 4: Replace the old 16 values with compact IDs plus three scalars**

Use `0` for N/A categorical states, shift `SpecialConditionType` values by one,
and encode matchup states as `0=N/A`, `1=false`, `2=true`. Compute matchup only
for `OptionType.ATTACK` and derive the attacking Pokemon type from
`CARD_ENERGY_TYPE_OFFSET`, not `CARD_TYPE_OFFSET`.

- [ ] **Step 5: Run focused tests**

Run: `pytest imitation_learning/tests/test_option_location_features.py -q`

Expected: PASS.

### Task 2: Update cache schema and training signatures

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/training/train.py`

**Interfaces:**
- Consumes: Task 1's 11 categorical IDs and 3 scalars per option.
- Produces: cache schema version 7 and decoder layout `option-location-auxiliary-v2`.

- [ ] **Step 1: Add a cache contract assertion test**

Extend the focused test module to assert:

```python
assert CACHE_SCHEMA_VERSION == 7
assert CACHE_OPTION_CATEGORICAL_DIM == 11
assert CACHE_OPTION_NUMERIC_DIM == 3
```

- [ ] **Step 2: Run the assertion and verify it fails against schema 6**

Run: `pytest imitation_learning/tests/test_option_location_features.py -q`

Expected: FAIL because the current cache schema is 6 with widths 5 and 16.

- [ ] **Step 3: Update packed-cache constants**

Set `CACHE_SCHEMA_VERSION = 7`, `OPTION_CATEGORICAL_DIM = 11`, and
`OPTION_NUMERIC_DIM = 3`. Keep unsigned 16-bit categorical storage: all location
and state IDs fit safely.

- [ ] **Step 4: Update both feature signatures identically**

```python
"decoder_layout": "option-location-auxiliary-v2",
"option_categorical_dim": 11,
"option_numeric_dim": 3,
"option_auxiliary_dim": 28,
```

- [ ] **Step 5: Run focused tests**

Run: `pytest imitation_learning/tests/test_option_location_features.py -q`

Expected: PASS.

### Task 3: Add shared location embeddings and one-hot auxiliary projection

**Files:**
- Modify: `imitation_learning/model/network.py`
- Extend: `imitation_learning/tests/test_option_location_features.py`

**Interfaces:**
- Consumes: Task 1 categorical columns 5-10 and numeric columns 0-2.
- Produces: `ImitationPolicy.encoder_location_embedding` with 21 rows, where row 20 is fixed zero padding.

- [ ] **Step 1: Add failing model tests**

```python
def test_encoder_and_option_share_location_table():
    model = make_tiny_policy()
    assert model.encoder_location_embedding.num_embeddings == 21
    assert model.encoder_location_embedding.padding_idx == 20


def test_auxiliary_expansion_has_28_values():
    model = make_tiny_policy()
    auxiliary = model.expand_option_auxiliary(categorical_row(), numeric_row())
    assert auxiliary.shape == (1, 28)
    assert auxiliary[0, 2:15].sum() == 1
    assert auxiliary[0, 15:21].sum() == 1
```

- [ ] **Step 2: Run the model tests and verify they fail**

Run: `pytest imitation_learning/tests/test_option_location_features.py -q`

Expected: FAIL because the shared location table and expansion helper do not exist.

- [ ] **Step 3: Implement shared location addition**

Create `Embedding(21, d_model, padding_idx=20)`. Add rows 0-19 to the reshaped
encoder tokens before the Transformer encoder. In `encode_options`, add rows from
categorical columns 5 and 6 directly to each option embedding.

- [ ] **Step 4: Implement exact 28-value one-hot expansion**

```python
auxiliary = torch.cat(
    (
        numeric[:, 0:2],
        F.one_hot(categorical[:, 7], 13).to(numeric.dtype),
        F.one_hot(categorical[:, 8], 6).to(numeric.dtype),
        numeric[:, 2:3],
        F.one_hot(categorical[:, 9], 3).to(numeric.dtype),
        F.one_hot(categorical[:, 10], 3).to(numeric.dtype),
    ),
    dim=1,
)
```

Replace `Linear(16, d_model)` with `Linear(28, d_model)`.

- [ ] **Step 5: Run focused tests**

Run: `pytest imitation_learning/tests/test_option_location_features.py -q`

Expected: PASS.

### Task 4: Synchronize the Kaggle submission notebook

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`

**Interfaces:**
- Consumes: Tasks 1-3 feature order, constants, location resolver, and model layers.
- Produces: inference code compatible with schema-7 checkpoints.

- [ ] **Step 1: Update notebook constants and model code**

Mirror widths 11/3/28, the 20 location IDs plus zero row, encoder location
addition, option location addition, and exact one-hot concatenation order.

- [ ] **Step 2: Update notebook feature extraction**

Mirror candidate/target location resolution, special-condition shifting,
attack-only matchup states, energy-type matchup calculation, removed fields,
and the `(number, count, attack_damage)` scalar order.

- [ ] **Step 3: Validate notebook JSON and remove stale contract text**

Run a JSON parse of the notebook and search its code cells for stale values:

```text
OPTION_NUMERIC_DIM = 16
option position
toolIndex / 4
energyIndex / 10
card_type
```

Expected: notebook parses successfully and no stale feature-building logic remains.

### Task 5: Final integration verification

**Files:**
- Verify: all files modified in Tasks 1-4

**Interfaces:**
- Produces: one consistent training/cache/submission contract.

- [ ] **Step 1: Run the focused test suite**

Run: `pytest imitation_learning/tests/test_option_location_features.py -q`

Expected: PASS.

- [ ] **Step 2: Run source and notebook consistency checks**

Confirm all Python modules use categorical width 11, stored numeric width 3,
auxiliary width 28, cache schema 7, and the same decoder-layout signature.

- [ ] **Step 3: Confirm migration behavior**

Verify that old feature caches fail with the schema-version message, while
winner-only extracted JSONL files remain accepted by `cache_features.py`.

- [ ] **Step 4: Review the final diff**

Run: `git diff --check` and inspect only files in this plan for unintended edits.
