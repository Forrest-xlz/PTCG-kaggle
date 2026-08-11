# Opponent Public-Action History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent three-step opponent public-action history token derived only from `Observation.logs`.

**Architecture:** Parse opponent logs into macro actions whose public event rows are pooled inside each action. Pack the latest three actions per episode and player perspective, encode them with opponent-only parameters, and append one optional token to the main encoder. Reproduce the parser and rolling state in the Kaggle notebook without using replay-only opponent selections.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, PyYAML, mmap packed cache, Kaggle notebook JSON, pytest.

## Global Constraints

- Preserve the existing own-action history implementation.
- Use only `Observation.logs` for opponent history.
- Never recover hidden Card IDs from replay-only information.
- Opponent and own history parameters must not share objects.
- Keep exactly three chronological opponent macro actions.
- Old checkpoints default `opponent_history_encoding` to `off`.
- Bump the cache schema; require recache but not re-extraction.

---

### Task 1: Observable Opponent Macro-Action Features

**Files:**
- Modify: `imitation_learning/model/features.py`
- Test: `imitation_learning/tests/test_opponent_history_features.py`

**Interfaces:**
- Produces: `OpponentHistoryActionFeatures`, `opponent_history_actions(obs, card_count, attack_count, numeric_catalog=None)`, opponent event-width constants, and unknown entity indices.
- Consumes later: cache packing in Task 2 and Kaggle parity in Task 4.

- [ ] **Step 1: Write failing aggregation tests**

Create typed/dict log fixtures proving that `ATTACK + HP_CHANGE + MOVE_CARD` produces one action, a following `PLAY` starts another action, current-player logs are excluded, and an orphan observable effect becomes `OTHER_PUBLIC_EFFECT`.

```python
actions = opponent_history_actions(obs, card_count=128, attack_count=32)
assert [action.action_type for action in actions] == [ATTACK_ACTION, PLAY_ACTION]
assert actions[0].event_categorical.shape[0] == 3
```

- [ ] **Step 2: Write failing unknown/public-entity tests**

Assert that absent and out-of-range Card/Attack/Area fields map to explicit unknown indices, while public `cardId`, `cardIdTarget`, and `attackId` remain available. Assert missing serial lookup yields zero Pokemon dynamics.

- [ ] **Step 3: Run the focused tests and confirm failure**

Run:

```powershell
pytest imitation_learning/tests/test_opponent_history_features.py -q
```

Expected: import failures for the new types and parser.

- [ ] **Step 4: Implement the parser and feature dataclass**

Add constants for macro-action, log-event, area, categorical, and numeric vocabularies. Represent each event row with:

```text
log_type, source_card_id, target_card_id, attack_id,
source_area, target_area, player_relation, special_condition
```

Add observable numeric/state values in a fixed float vector. Use existing card, attack, and Pokemon dynamic conventions; locate public Pokemon by serial across Active and Bench and return zeros when absent.

- [ ] **Step 5: Run focused and existing feature tests**

Run the new test plus `test_action_history_features.py` and `test_option_features.py`. Existing own-history outputs must be unchanged.

### Task 2: Packed Cache and Perspective-Safe Rolling History

**Files:**
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/training/feature_cache.py`
- Test: `imitation_learning/tests/test_opponent_history_cache.py`
- Test: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- Produces: packed opponent action type/valid/event categorical/event numeric/Pokemon dynamic/attack dynamic/event-offset sections in `FeatureRecord`, `FeatureView`, and `CachedBatch`.
- Consumes: `opponent_history_actions` from Task 1.

- [ ] **Step 1: Write failing packed-record tests**

Build records with zero, one, and three opponent macro actions. Assert left padding, event boundaries, dtype/range validation, shard round-trip, batch collation, and cache rejection for malformed boundaries.

- [ ] **Step 2: Write failing chronology/isolation tests**

Feed interleaved records for players 0 and 1 across two episodes. Assert visible opponent logs are appended before the current sample is packed and that histories do not cross player perspectives or episode boundaries.

- [ ] **Step 3: Run cache tests and confirm failure**

Run:

```powershell
pytest imitation_learning/tests/test_opponent_history_cache.py imitation_learning/tests/test_feature_cache.py -q
```

- [ ] **Step 4: Extend packed schema and cache writer/reader**

Increment `CACHE_SCHEMA_VERSION`. Add fixed arrays for three action slots and ragged event arrays with offset pointers, mirroring existing own-history storage while using opponent-specific widths. Extend validation, mmap views, batching, and tensor conversion.

- [ ] **Step 5: Maintain per-perspective deques during cache creation**

Use `opponent_histories: dict[int, deque[OpponentHistoryActionFeatures]]`. For each record, parse its opponent logs and extend that player's deque before calling `_prepare_record`; clear the mapping on episode changes.

- [ ] **Step 6: Run cache regressions**

Run opponent cache, existing feature cache, history cache, and training batch tests.

### Task 3: Independent Opponent Temporal Encoder Token

**Files:**
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/model/network.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/validation/evaluate.py`
- Test: `imitation_learning/tests/test_opponent_history_network.py`
- Test: `imitation_learning/tests/test_validation_config.py`

**Interfaces:**
- Adds `ModelConfig.opponent_history_encoding`, `opponent_history_action_mlp_layers`, and `opponent_history_sequence_mlp_layers`.
- Extends model forward inputs with the packed opponent history tensors.

- [ ] **Step 1: Write failing config and independence tests**

Assert valid modes, layer constraints, YAML parsing, checkpoint round-trip, old-config default `off`, and that every opponent embedding/projection/MLP parameter object differs from its own-history counterpart.

- [ ] **Step 2: Write failing mode/token/mask tests**

Assert `off` adds no token, enabled modes add exactly one token, an empty three-slot history is padding-masked, and `basic`, `structural`, and `full` consume only their intended fields.

- [ ] **Step 3: Run network tests and confirm failure**

Run:

```powershell
pytest imitation_learning/tests/test_opponent_history_network.py imitation_learning/tests/test_validation_config.py -q
```

- [ ] **Step 4: Implement opponent-only encoder modules**

Create independent macro/log/area/entity embeddings, static/dynamic projections, action MLP, and sequence MLP. Pool event embeddings per macro action, left-to-right concatenate three action embeddings, produce one token, and extend the encoder padding mask.

- [ ] **Step 5: Wire training and standalone validation**

Pass opponent arrays through cache-to-tensor conversion, training, evaluation, checkpoint architecture metadata, feature signatures, and ensemble compatibility checks. Add the three YAML values with concise comments.

- [ ] **Step 6: Run model/training/validation regressions**

Run opponent network tests plus existing action-history network, training schedule/metrics, validation config, and standalone evaluate tests.

### Task 4: Kaggle Agent Parity and Documentation

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Modify: `imitation_learning/README.md`
- Test: `imitation_learning/tests/test_submission_opponent_history.py`

**Interfaces:**
- Produces: generated `main.py` with `OPPONENT_ACTION_HISTORY`, log aggregation parity, unknown mapping, checkpoint-driven configuration, and model inputs matching training.

- [ ] **Step 1: Write failing notebook artifact tests**

Assert generated source contains the opponent deque, clears it at new-game initialization, parses logs before inference, passes all opponent tensors, defaults absent checkpoint fields to `off`, and never reads the other replay player's selected option.

- [ ] **Step 2: Run artifact test and confirm failure**

Run:

```powershell
pytest imitation_learning/tests/test_submission_opponent_history.py -q
```

- [ ] **Step 3: Patch notebook cells with production-equivalent logic**

Mirror Task 1 aggregation and Task 3 model encoding in the generated agent. Preserve deck selection, ensembles, expert flag behavior present on the branch, model loading, and submission packaging.

- [ ] **Step 4: Document rebuild and compatibility behavior**

Document independent opponent-history settings, public-log information limits, no-reextract/full-recache requirement, old-checkpoint `off` fallback, and new-checkpoint training requirement.

- [ ] **Step 5: Run final verification**

Run all focused tests and relevant regressions; compile modified Python; parse notebook JSON and compile its generated `main.py`; load YAML; run `git diff --check`; inspect modified-file scope. Do not claim the unrelated stale test suite is green if baseline tests outside this feature still fail.
