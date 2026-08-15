# Revealed-Hand and Learnable CLS Tokens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two stateful revealed-hand encoder tokens and an optional learnable CLS encoder token across cache generation, training, validation, export, and Kaggle inference.

**Architecture:** A small perspective-local tracker consumes replay/API logs and maintains exact known hand cards by serial. Cache generation pools those cards into two new sparse encoder words; the model gives the words independent learned Card ID ranges, applies separate equal-depth token MLPs, padding-masks empty words, and optionally appends a learned CLS vector.

**Tech Stack:** Python 3.10, NumPy, PyTorch, mmap feature caches, PyYAML, Jupyter notebook JSON.

## Global Constraints

- Do not change replay extraction, action enumeration, policy targets, validation splits, or decoder behavior.
- Existing extracted JSONL data must remain usable; packed feature caches must be rebuilt.
- Revealed-hand identity is tracked by card serial and represented by Card ID counts.
- Own and opponent revealed-hand learned embeddings and token MLP weights must not be shared.
- Static 54-dimensional card projections follow the existing `card_mlp_scope` behavior.
- The CLS token is controlled only by `learnable_cls_token` and is never padding-masked.
- The maximum encoder sequence is 30 tokens: 28 base, one optional history token, and one optional CLS token.

---

### Task 1: Public revealed-hand state tracker

**Files:**
- Create: `imitation_learning/model/revealed_hand.py`
- Create: `imitation_learning/tests/test_revealed_hand.py`

**Interfaces:**
- Produces: `RevealedHandTracker.reset()`, `RevealedHandTracker.update(logs)`, and `RevealedHandTracker.relative_cards(your_index)`.
- `relative_cards(your_index)` returns `(own_card_ids, opponent_card_ids)` with duplicate copies preserved.

- [ ] **Step 1: Write failing tracker tests**

Use `types.SimpleNamespace` logs to verify literal behavior:

```python
def log(type_, player=0, card=None, serial=None, source=None, target=None):
    return SimpleNamespace(
        type=type_, playerIndex=player, cardId=card, serial=serial,
        fromArea=source, toArea=target,
    )

def test_public_cards_enter_and_leave_known_hand():
    tracker = RevealedHandTracker()
    tracker.update([log(6, card=741, serial=10, source=12, target=2)])
    assert tracker.relative_cards(0) == ([741], [])
    tracker.update([log(10, card=741, serial=10)])
    assert tracker.relative_cards(0) == ([], [])

def test_duplicate_card_ids_are_preserved_by_serial():
    tracker = RevealedHandTracker()
    tracker.update([
        log(6, card=741, serial=10, source=12, target=2),
        log(6, card=741, serial=11, source=12, target=2),
    ])
    assert tracker.relative_cards(0) == ([741, 741], [])

def test_private_draw_does_not_reveal_card():
    tracker = RevealedHandTracker()
    tracker.update([log(4, card=741, serial=10)])
    assert tracker.relative_cards(0) == ([], [])

def test_face_down_hand_departure_clears_player_knowledge():
    tracker = RevealedHandTracker()
    tracker.update([log(6, player=1, card=305, serial=20, source=12, target=2)])
    tracker.update([log(7, player=1, source=2, target=1)])
    assert tracker.relative_cards(0) == ([], [])
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_revealed_hand.py -q
```

Expected: collection fails because `model.revealed_hand` does not exist.

- [ ] **Step 3: Implement the tracker**

Implement a two-player dictionary keyed by serial. Treat log types 6/10/11/12 and area value 2 as the stable API contract; ignore malformed additions, make unknown removals no-ops, and clear only the affected player on ambiguous face-down movement out of hand.

- [ ] **Step 4: Run tracker tests**

Run the Task 1 pytest command. Expected: all tracker tests pass.

- [ ] **Step 5: Commit the tracker**

```powershell
git add imitation_learning/model/revealed_hand.py imitation_learning/tests/test_revealed_hand.py
git commit -m "feat: track publicly revealed hand cards"
```

### Task 2: Feature layout and packed cache integration

**Files:**
- Modify: `imitation_learning/model/features.py`
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/training/feature_cache.py`
- Create: `imitation_learning/tests/test_revealed_hand_features.py`

**Interfaces:**
- `encoder_features(..., own_revealed_hand=(), opponent_revealed_hand=())` produces 28 sparse words.
- `FeatureRecord.revealed_hand_present` and `CachedBatch.revealed_hand_present` have width 2 and values in `{0, 1}`.
- Cache schema becomes 17 with encoder layout `numeric-summary-28-revealed-hand-v1`.

- [ ] **Step 1: Write failing encoder-layout tests**

Construct the smallest typed observation fixture already accepted by
`encoder_features`, pass `[741, 741]` and `[305]`, and assert:

```python
assert len(encoded.sparse.offset) == 28
assert encoded.revealed_hand_present == [1, 1]
assert encoded.sparse.index[-3:] == [own_revealed_base + 741,
                                     own_revealed_base + 741,
                                     opponent_revealed_base + 305]
```

Also assert empty inputs produce `[0, 0]` and two empty words.

- [ ] **Step 2: Run the new test and verify it fails on the 26-word layout**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_revealed_hand_features.py -q
```

Expected: failure because the encoder has 26 words and no presence field.

- [ ] **Step 3: Extend pure feature construction**

Raise `ENCODER_TOKENS` to 28, add two card ranges after the current global-summary placeholder, and return the two presence flags. Use weight `1.0` so duplicates contribute by count.

- [ ] **Step 4: Feed stateful cards into cache records**

Maintain one `RevealedHandTracker` per replay perspective next to the existing action-history deques. Reset trackers on episode change, update from `obs.logs` before encoding, and pass perspective-relative cards into `encoder_features`.

- [ ] **Step 5: Persist presence flags and invalidate old caches**

Add a `revealed_hand_present` uint8 section to writer, shard, view, collate, and batch types. Validate shape and binary values. Change `CACHE_SCHEMA_VERSION` from 16 to 17, `ENCODER_WORDS` from 26 to 28, and the feature signature layout string.

- [ ] **Step 6: Run feature/cache tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_revealed_hand.py imitation_learning/tests/test_revealed_hand_features.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit feature-cache changes**

```powershell
git add imitation_learning/model/features.py imitation_learning/training/cache_features.py imitation_learning/training/feature_cache.py imitation_learning/tests/test_revealed_hand_features.py
git commit -m "feat: cache revealed hand encoder tokens"
```

### Task 3: Model, configuration, training, and validation

**Files:**
- Modify: `imitation_learning/model/network.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/validation/metrics.py`
- Modify: `imitation_learning/cfg/train.yaml`
- Create: `imitation_learning/tests/test_revealed_hand_model.py`

**Interfaces:**
- `ModelConfig.revealed_hand_token_mlp_layers: int = 0`.
- `ModelConfig.learnable_cls_token: bool = False`.
- Model forward adds `revealed_hand_present` immediately after the three dense summaries.
- Encoder mask width exactly equals `28 + history_enabled + cls_enabled`.

- [ ] **Step 1: Write failing model tests**

Build a tiny model (`d_model=8`, one head/layer, zero dropout) and assert:

```python
assert model.encoder_token_count == 28
assert model.own_revealed_hand_token_mlp is not model.opponent_revealed_hand_token_mlp

config = replace(config, learnable_cls_token=True)
model = make_model(config)
assert model.encoder_token_count == 29
assert tuple(model.cls_token.shape) == (1, 1, 8)

mask = model._encoder_padding_mask(own, opponent, revealed, history_valid=None)
assert mask[:, 26:28].tolist() == [[False, True]]
assert mask.shape[1] == 29  # includes the enabled CLS token
```

Also assert negative revealed-hand MLP layers and non-Boolean CLS values raise
`ValueError`.

- [ ] **Step 2: Run tests and verify missing configuration failures**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_revealed_hand_model.py -q
```

Expected: failure because the two configuration fields do not exist.

- [ ] **Step 3: Extend encoder vocabulary mappings and token MLPs**

Map the two added sparse ranges to `own_hand` and `opponent_hand` static regions,
which reuses existing 54-dimensional region projection parameters while their
disjoint embedding-bag rows provide independent learned Card ID embeddings.
Create independent token MLP modules at the configured common depth and apply
them to positions 26 and 27.

- [ ] **Step 4: Add CLS assembly and masks**

Create `cls_token` only when enabled, initialize it with the same small normal
initialization convention used for learned model tokens, append it after the
optional history token, and append a `False` mask column. Mask revealed-hand
positions from `revealed_hand_present`.

- [ ] **Step 5: Thread the batch field through train and validation**

Update every model invocation to move `batch.revealed_hand_present` to the model
device as `torch.long`. Extend `ModelSettings`, `ModelConfig` construction, and
the YAML with:

```yaml
revealed_hand_token_mlp_layers: 1  # Separate own/opponent token MLP weights.
learnable_cls_token: true          # Append one global learned encoder token.
```

- [ ] **Step 6: Run model and integration tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit model and runtime changes**

```powershell
git add imitation_learning/model/network.py imitation_learning/training/train.py imitation_learning/validation/metrics.py imitation_learning/cfg/train.yaml imitation_learning/tests/test_revealed_hand_model.py
git commit -m "feat: encode revealed hands and optional cls token"
```

### Task 4: Kaggle submission parity and final verification

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Modify: `imitation_learning/README.md`
- Create: `imitation_learning/tests/test_submission_notebook.py`

**Interfaces:**
- Notebook `ModelConfig` accepts saved checkpoint fields without manual architecture edits.
- Notebook agent maintains one revealed-hand tracker for the current match and clears it wherever action history is cleared.
- Notebook feature tensors and model argument order match training exactly.

- [ ] **Step 1: Write a failing notebook execution/parity test**

Load notebook code cells into one namespace up to (but not invoking) the live
agent, construct synthetic logs, and assert the notebook tracker and project
tracker return the same relative Card ID lists. Instantiate a tiny checkpoint
architecture with CLS both disabled and enabled and assert strict state-dict
loading succeeds for each matching model.

- [ ] **Step 2: Run notebook test and verify it fails before parity code exists**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_submission_notebook.py -q
```

Expected: failure because notebook configuration and tracker fields are absent.

- [ ] **Step 3: Mirror feature and model changes in the notebook**

Update constants, sparse layout, mappings, model configuration, MLPs, CLS
assembly, masks, and forward arguments. Add the tracker to agent state, apply
logs before feature construction, and reset it at the same match boundary as
`ACTION_HISTORY`.

- [ ] **Step 4: Document cache rebuild and configuration**

State that users run only `training/cache_features.py` again, not extraction,
and describe both new model parameters and the resulting token counts.

- [ ] **Step 5: Run complete verification**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests -q
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m compileall -q imitation_learning/model imitation_learning/training imitation_learning/validation
git diff --check
```

Expected: pytest exits with zero failures, compileall exits 0, and diff check
prints no errors.

- [ ] **Step 6: Commit notebook and documentation**

```powershell
git add imitation_learning/kaggle_submission_imitation_agent.ipynb imitation_learning/README.md imitation_learning/tests/test_submission_notebook.py
git commit -m "feat: add revealed hand inference parity"
```

