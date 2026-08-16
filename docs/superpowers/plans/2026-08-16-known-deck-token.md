# Known-Deck Token Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stateful encoder token containing the physical cards currently known to be inside the acting player's deck.

**Architecture:** A reusable log tracker maintains `serial -> card_id` from public engine events. Cache construction and Kaggle inference own tracker lifecycle, while feature construction only receives current Card IDs and creates the 27th sparse token. The network assigns this region an independent learned embedding and an optional region-token MLP.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, PyYAML, pytest, Jupyter notebook JSON

## Global Constraints

- Do not change extracted JSONL schema or require re-extraction.
- Increment packed cache schema and require re-caching.
- Track only the acting player's own cards.
- Key physical cards by unique `serial`; repeated events must be idempotent.
- Preserve all existing model and training behavior outside the new token.

---

### Task 1: Known-Deck Log Tracker

**Files:**
- Create: `imitation_learning/model/known_deck.py`
- Create: `imitation_learning/tests/test_known_deck_tracker.py`

**Interfaces:**
- Produces: `KnownDeckTracker.reset()`, `KnownDeckTracker.update(logs, player_index)`, and `KnownDeckTracker.card_ids() -> tuple[int, ...]`
- Consumes: raw dict logs or typed `cg.api.Log` objects, using `LogType.DRAW`, `LogType.MOVE_CARD`, and `AreaType.DECK` numeric values

- [ ] **Step 1: Write failing tracker tests**

Cover own move-to-deck add, duplicate add idempotence, own draw removal,
move-from-deck removal, shuffle retention, opponent isolation, two serials with
the same Card ID, malformed event ignore, and reset.

- [ ] **Step 2: Run tests and verify missing-module failure**

```powershell
python -m pytest imitation_learning/tests/test_known_deck_tracker.py -q
```

- [ ] **Step 3: Implement the tracker**

Normalize dict/dataclass fields through a small accessor. Insert only when
both `serial` and `cardId` are valid integers. Remove by serial for own draw or
any own move whose `fromArea` is deck. Process removal before insertion so a
deck-to-deck event remains known.

- [ ] **Step 4: Run tracker tests and commit**

```powershell
git add -f imitation_learning/tests/test_known_deck_tracker.py
git add imitation_learning/model/known_deck.py
git commit -m "feat: track known cards in own deck"
```

### Task 2: Feature and Packed Cache Integration

**Files:**
- Modify: `imitation_learning/model/features.py`
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/tests/test_numeric_summary_features.py`
- Modify: `imitation_learning/tests/test_feature_cache.py`
- Create: `imitation_learning/tests/test_known_deck_cache.py`

**Interfaces:**
- Consumes: `encoder_features(..., known_deck_card_ids=())`
- Produces: 27 sparse encoder offsets; cache schema version incremented by one
- Uses: one `KnownDeckTracker` per `(episode, player)` during sequential source processing

- [ ] **Step 1: Write failing feature/cache tests**

Assert the new encoder token contains one weighted entry per supplied Card ID,
including two copies of the same ID; an empty list has an empty token. Assert
cache records use 27 offsets. Exercise two sequential records where a move-to-
deck log adds a card, a repeated log does not duplicate it, and a later draw
removes it. Ensure tracker update occurs before `_prepare_record`, including
when preparation returns a skip reason.

- [ ] **Step 2: Run focused tests and verify failure**

```powershell
python -m pytest imitation_learning/tests/test_numeric_summary_features.py imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_known_deck_cache.py -q
```

- [ ] **Step 3: Add the 27th sparse token**

Set encoder token/word constants to 27. Add one known-deck vocabulary range
after the full-deck range. In `encoder_features`, append a token that adds every
current known Card ID with weight `1.0`, then advances by `card_count`.

- [ ] **Step 4: Integrate cache lifecycle**

Increment cache schema. In each source worker, clear trackers on episode
change, update the acting player's tracker from raw observation logs before
preparing a record, and pass `tracker.card_ids()` into feature construction.
Do not modify extraction metadata requirements.

- [ ] **Step 5: Run focused tests and commit**

```powershell
git add imitation_learning/model/features.py imitation_learning/training/cache_features.py imitation_learning/training/feature_cache.py imitation_learning/tests/test_numeric_summary_features.py imitation_learning/tests/test_feature_cache.py
git add -f imitation_learning/tests/test_known_deck_cache.py
git commit -m "feat: cache known-deck encoder token"
```

### Task 3: Network and Training Configuration

**Files:**
- Modify: `imitation_learning/model/network.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/tests/test_region_card_projection.py`
- Modify: `imitation_learning/tests/test_artifacts.py`
- Create: `imitation_learning/tests/test_known_deck_network.py`

**Interfaces:**
- Produces: `ModelConfig.known_deck_token_mlp_layers: int = 0`
- Consumes: YAML `model.known_deck_token_mlp_layers`
- Token contribution: independent `own_known_deck` learned Card ID embedding plus static-card projection, then optional token MLP

- [ ] **Step 1: Write failing network/config tests**

Assert `ENCODER_TOKENS == 27`, the new vocabulary range maps to Card IDs and
the `own_known_deck` region, shared/region static projection both work, MLP
layer zero preserves the sum, positive layers transform it, and
`region_token_mlp_residual` controls replacement versus addition. Assert YAML
and `ModelSettings` contain the new field and reject negative layers.

- [ ] **Step 2: Run focused tests and verify failure**

Run the new network test with the PyTorch environment and the pure artifact
test with the base environment:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' imitation_learning/tests/test_known_deck_network.py
& 'C:\Users\Liuluotu\Anaconda\anaconda3\python.exe' -m pytest imitation_learning/tests/test_artifacts.py -q
```

Expected: failures for the missing config field, 27th token, and region.

- [ ] **Step 3: Implement network/config changes**

Add `own_known_deck` to card regions and encoder mappings. Increase fixed token
count to 27. Instantiate `own_known_deck_token_mlp`, apply it to the new token
without changing other slices, validate a nonnegative layer count, and pass
the YAML field through `ModelSettings` into `ModelConfig`.

- [ ] **Step 4: Run focused tests and commit**

Repeat the Step 2 commands and expect both to pass, then commit:

```powershell
git add imitation_learning/model/network.py imitation_learning/training/train.py imitation_learning/cfg/train.yaml imitation_learning/tests/test_region_card_projection.py imitation_learning/tests/test_artifacts.py
git add -f imitation_learning/tests/test_known_deck_network.py
git commit -m "feat: encode known-deck token in policy network"
```

### Task 4: Kaggle Submission Parity

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Create: `imitation_learning/tests/test_submission_known_deck.py`

**Interfaces:**
- Consumes: checkpoint `model_config.known_deck_token_mlp_layers`
- Produces: generated `main.py` with one global `KnownDeckTracker`, reset at initial deck selection and update before encoder construction

- [ ] **Step 1: Write failing notebook-source tests**

Extract the `%%writefile main.py` cell and assert it contains the same serial-
keyed transitions, reset call, pre-encoding update, 27-token constants, new
region mapping, MLP config, and checkpoint-driven architecture. Compile the
generated source without importing the engine.

- [ ] **Step 2: Run notebook test and verify failure**

```powershell
python -m pytest imitation_learning/tests/test_submission_known_deck.py -q
```

- [ ] **Step 3: Patch notebook source**

Update only the generated-agent cell and architecture configuration. Keep deck
selection, ensemble behavior, action history, and inference policy unchanged.

- [ ] **Step 4: Run notebook and regression tests**

Run the new notebook test plus existing artifact/submission tests. Compile all
modified Python files and run `git diff --check`.

- [ ] **Step 5: Commit notebook parity**

```powershell
git add imitation_learning/kaggle_submission_imitation_agent.ipynb
git add -f imitation_learning/tests/test_submission_known_deck.py
git commit -m "feat: maintain known-deck token in Kaggle agent"
```
