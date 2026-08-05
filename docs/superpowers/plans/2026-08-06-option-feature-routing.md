# Routed Option Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the decoder's mixed 76-dimensional option vector with categorical embeddings, routed Pokemon/attack dynamics, and a configurable post-sum option-token MLP.

**Architecture:** `model.features.decoder_features` will emit an 11-column categorical matrix plus separate 46-dimensional Pokemon and 6-dimensional attack dynamic matrices. `PolicyNetwork` will embed categorical fields, project both dynamic groups, mask absent groups to exact zero, sum all branches, and optionally transform each option token before existing composite-action summation and decoder cross-attention.

**Tech Stack:** Python 3.10, NumPy, PyTorch, YAML, packed mmap feature cache, Jupyter notebook JSON.

## Global Constraints

- Keep the existing encoder, composite-action sum, and decoder cross-attention behavior unchanged.
- `option_token_mlp_layers=0` is identity; positive depths follow the existing `Linear`, then repeated `ReLU -> Linear`, convention.
- Missing Pokemon and attack dynamic projections must be exactly zero after projection bias.
- Old feature caches must fail schema validation; replay extraction files remain reusable.
- Inference architecture is reconstructed from checkpoint configuration.

---

### Task 1: Feature Routing Contract

**Files:**
- Modify: `imitation_learning/model/features.py`
- Create: `imitation_learning/tests/test_option_features.py`

**Interfaces:**
- Produces: `OptionFeatures(categorical, pokemon_dynamic, attack_dynamic, action_index, action_offset)`.
- Produces: `OPTION_CATEGORICAL_DIM=11`, `POKEMON_DYNAMIC_DIM=46`, and `ATTACK_DYNAMIC_DIM=6`.
- Consumes: engine `Observation`, static `NumericFeatureCatalog`, and legal action memberships.

- [ ] **Step 1: Add focused tests for categorical encoding, routing, missing slots, and attack-derived values**

Create fixtures with minimal observation objects and assert that `number`, `count`, player relation, areas, and special condition occupy distinct categorical columns; assert ATTACK routes own/opponent Active Pokemon; assert ATTACH routes only its in-play target; assert absent slots are zero.

- [ ] **Step 2: Run the feature tests and verify the old representation fails them**

Run: `python -m pytest imitation_learning/tests/test_option_features.py -q`

Expected: failures because `OptionFeatures` has only `numeric` and categorical width five.

- [ ] **Step 3: Replace the mixed numeric builder with categorical and routed dynamic builders**

Implement explicit option-value encoding with missing index 0, exact values 0..60 at indices 1..61, and overflow index 62. Resolve primary and secondary Pokemon per the approved routing table. Build two 23-value Pokemon blocks, concatenate them, and build six attack-dynamic values only for valid ATTACK options.

- [ ] **Step 4: Run the focused feature tests**

Run: `python -m pytest imitation_learning/tests/test_option_features.py -q`

Expected: all feature-routing tests pass.

### Task 2: Packed Cache Schema

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/training/cache_features.py`
- Create: `imitation_learning/tests/test_option_cache.py`

**Interfaces:**
- Consumes: the new `OptionFeatures` arrays.
- Produces: packed cache schema 13 with `option_categorical`, `pokemon_dynamic`, and `attack_dynamic` float16 arrays.
- Produces: dataset and batch attributes with the same three names.

- [ ] **Step 1: Add cache round-trip and previous-schema rejection tests**

Construct a small `FeatureRecord`, write one packed shard, reload it, and compare all three option arrays and action membership. Assert schema 12 metadata is rejected.

- [ ] **Step 2: Run cache tests and verify failure against schema 12**

Run: `python -m pytest imitation_learning/tests/test_option_cache.py -q`

Expected: failures for missing dynamic fields and unchanged schema version.

- [ ] **Step 3: Update records, buffers, files, loader slices, collator, and signatures**

Replace every `option_numeric` path with separate Pokemon and attack dynamic paths, validate their lengths against option count, narrow them safely to float16, and bump `CACHE_SCHEMA_VERSION` to 13.

- [ ] **Step 4: Run cache tests**

Run: `python -m pytest imitation_learning/tests/test_option_cache.py -q`

Expected: all cache tests pass.

### Task 3: Policy Network and Configuration

**Files:**
- Modify: `imitation_learning/model/network.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/README.md`
- Create: `imitation_learning/tests/test_option_network.py`

**Interfaces:**
- Consumes: categorical `(N,11)`, Pokemon dynamic `(N,46)`, and attack dynamic `(N,6)` tensors.
- Produces: one `d_model` option token per option.
- Configures: `ModelConfig.option_token_mlp_layers: int >= 0`.

- [ ] **Step 1: Add network tests for masking and MLP depths**

Assert absent Pokemon/attack groups contribute exact zeros despite projection bias; assert depth zero is identity after branch summation; assert depths one and two construct the documented modules and preserve tensor shapes.

- [ ] **Step 2: Run network tests and verify failure**

Run: `python -m pytest imitation_learning/tests/test_option_network.py -q`

Expected: failures because the model still accepts a 76-dimensional numeric tensor.

- [ ] **Step 3: Add categorical embeddings, dynamic projections, masks, and post-sum MLP**

Remove `option_numeric_projection`. Add number, count, player relation, area, in-play area, and special-condition embeddings. Add `Linear(46,d_model)` and `Linear(6,d_model)` projections, apply presence masks after projection, sum all branches, then apply the optional option-token MLP before `combine_actions`.

- [ ] **Step 4: Replace training/config/documentation references**

Rename configuration to `option_token_mlp_layers`, validate `>=0`, pass new batch tensors to the model, update cache signatures, and document the new semantics and required cache rebuild.

- [ ] **Step 5: Run model and feature/cache tests**

Run: `python -m pytest imitation_learning/tests/test_option_features.py imitation_learning/tests/test_option_cache.py imitation_learning/tests/test_option_network.py -q`

Expected: all tests pass.

### Task 4: Kaggle Submission Notebook

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`

**Interfaces:**
- Consumes: checkpoint `config` containing `option_token_mlp_layers` and the new model weights.
- Produces: inference tensors matching training feature extraction exactly.

- [ ] **Step 1: Update embedded constants, model config, and model implementation**

Mirror the 11 categorical fields, new embeddings, two dynamic projections and masks, and post-sum option-token MLP. Remove every `OPTION_NUMERIC_DIM` and `option_numeric_mlp_layers` reference.

- [ ] **Step 2: Update embedded decoder feature extraction and agent call**

Mirror routing and normalization exactly, return both dynamic matrices, and pass them through the notebook model forward call.

- [ ] **Step 3: Validate notebook JSON and stale-name absence**

Run a JSON parser over the notebook and search for `option_numeric`, `OPTION_NUMERIC_DIM`, and `option_numeric_mlp_layers`.

Expected: valid notebook JSON and no stale references.

### Task 5: Final Verification

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: a clean, internally consistent architecture ready for cache rebuild and training.

- [ ] **Step 1: Run available automated tests and syntax checks**

Run the three focused pytest files and compile the modified Python modules. If Python is unavailable in the local environment, record that limitation and run JSON parsing, `git diff --check`, and exhaustive stale-symbol searches instead.

- [ ] **Step 2: Review the full diff against the approved design**

Confirm routing, normalization, masks, cache schema, configuration, checkpoint construction, and notebook inference use identical dimensions and field ordering.

- [ ] **Step 3: Report migration steps**

Tell the user that replay extraction does not need rerunning, feature cache must be rebuilt, and old checkpoints are intentionally incompatible because their architecture lacks the new embeddings and projections.
