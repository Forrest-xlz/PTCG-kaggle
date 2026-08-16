# Player Setup Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add observable whole-board Energy and formation-readiness features to the own and opponent player summaries.

**Architecture:** Extend the immutable numeric feature catalog with card evolution metadata and printed attack costs, then compute a deck-aware setup vector during feature caching. Own and opponent summaries share eleven public features, while only the own summary receives four hand-derived features. Packed caches and the network consume the new fixed widths; Kaggle inference duplicates the same deterministic calculations.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, PyYAML, pytest, Jupyter notebook JSON

## Global Constraints

- New own summary width is exactly 84; new opponent summary width is exactly 82.
- Never read opponent hidden hand contents for the new features.
- Use `Pokemon.energies` for effective Energy and `energyCards` for physical cards.
- Restrict evolution-chain relationships to the configured starting deck.
- Do not change extracted JSONL schema; increment packed cache schema and require re-caching.
- Keep training, standalone validation, and Kaggle inference feature semantics identical.
- Preserve all unrelated encoder, decoder, loss, sampling, and action-history behavior.

---

### Task 1: Static Setup Catalog and Energy-Cost Arithmetic

**Files:**
- Modify: `imitation_learning/model/features.py`
- Create: `imitation_learning/tests/test_setup_feature_catalog.py`

**Interfaces:**
- Produces: `SetupFeatureCatalog`, `_build_setup_feature_catalog(cards, attacks, card_count)`, and `_attack_energy_deficit(available, required) -> int`
- Extends: `NumericFeatureCatalog.setup: SetupFeatureCatalog`
- Consumes: `CardData.basic/stage1/stage2/name/evolvesFrom/cardType/energyType/skills/attacks` and `Attack.energies`

- [ ] **Step 1: Write failing catalog and Energy matching tests**

Create lightweight card and attack dataclasses in the test. Assert card-aligned
names, stages (`0/1/2`, `-1` for non-Pokemon), types, `evolvesFrom`, Basic
Energy flags, attack-cost lookup, and discovery of a Stadium whose skill text
contains the canonical same-turn evolution rule. Add separate assertions for
exact typed costs, Colorless payment, Rainbow wildcard payment, Team Rocket
Psychic/Darkness payment, surplus Energy, and unmatched deficits.

- [ ] **Step 2: Run the test and verify RED**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' imitation_learning/tests/test_setup_feature_catalog.py
```

Expected: import failure for `SetupFeatureCatalog` or missing helper functions.

- [ ] **Step 3: Implement the immutable setup catalog**

Add card-aligned tuples/arrays for names, stages, card types, Energy types,
evolution parents, Basic Energy flags, attack requirements, and same-turn
evolution Stadium Energy types. Build it once inside `_default_numeric_catalog`
and validate all card-aligned lengths in `__post_init__`.

- [ ] **Step 4: Implement deterministic Energy-cost matching**

Consume available units without mutation of the observation: satisfy typed
requirements first using exact, Rainbow, then Team Rocket compatibility; use
remaining units for Colorless; return the number of unmatched requirements.

- [ ] **Step 5: Run the focused test and commit**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' imitation_learning/tests/test_setup_feature_catalog.py
git add imitation_learning/model/features.py
git add -f imitation_learning/tests/test_setup_feature_catalog.py
git commit -m "feat: catalog setup and attack energy metadata"
```

### Task 2: Own and Opponent Setup Summaries

**Files:**
- Modify: `imitation_learning/model/features.py`
- Create: `imitation_learning/tests/test_player_setup_summary.py`

**Interfaces:**
- Produces: `_public_setup_summary(player, deck, catalog) -> list[float]` of length 11
- Produces: `_own_hand_setup_summary(state, player, deck, catalog) -> list[float]` of length 4
- Changes: `_player_summary(player, catalog, deck, state, include_private)` appends the correct vector

- [ ] **Step 1: Write failing public-summary tests**

Construct Active and Bench Pokemon containing different physical and effective
Energy counts. Assert normalized Energy totals, three stage counts,
deck-restricted incomplete and terminal evolution counts, attack-ready count,
total/minimum deficits, the no-attack sentinel, and `benchMax`-based empty
slots.

- [ ] **Step 2: Write failing own-hand and visibility tests**

Assert distinct immediately-evolvable Pokemon versus physical matching hand
cards, Basic Pokemon/Energy counts, `appearThisTurn`, `state.turn < 2`, and a
same-turn evolution Stadium. Pass an opponent object with `hand=None` and
assert no private feature function is called and no hidden values are added.

- [ ] **Step 3: Run both tests and verify RED**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' imitation_learning/tests/test_player_setup_summary.py
```

Expected: missing setup-summary helper failures.

- [ ] **Step 4: Implement deck-aware evolution helpers and summaries**

Cache the evolution graph with a hashable key containing `tuple(deck)`,
`catalog.card_names`, and `catalog.evolves_from`. Count a board Pokemon as
incomplete when its name has a direct child in the deck. Count it as terminal
only when it participates in a deck evolution line and has no child. Match
hand evolution cards by `evolvesFrom == current_card_name`.

- [ ] **Step 5: Append asymmetric summary features**

Append eleven public features to both players and four private hand features
only to the acting player. Update `OWN_SUMMARY_DIM` from 69 to 84 and
`OPPONENT_SUMMARY_DIM` from 71 to 82. Keep the original feature order intact
and append all new values at the end.

- [ ] **Step 6: Run tests and commit**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' imitation_learning/tests/test_player_setup_summary.py
git add imitation_learning/model/features.py
git add -f imitation_learning/tests/test_player_setup_summary.py
git commit -m "feat: add player setup summary features"
```

### Task 3: Cache, Network, Training, and Validation Widths

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/model/network.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/validation/evaluate.py`
- Modify: `imitation_learning/tests/test_feature_cache.py`
- Create: `imitation_learning/tests/test_setup_summary_integration.py`

**Interfaces:**
- Produces: packed schema version 18
- Produces: encoder feature layout `numeric-summary-27-known-deck-setup-v4`
- Consumes: own arrays of width 84 and opponent arrays of width 82

- [ ] **Step 1: Write failing width and signature tests**

Assert feature-cache validation accepts exactly 84/82, rejects 69/71, packed
metadata records schema 18, train/cache/evaluate signatures match, and
`PTCGTransformer` summary projections have input widths 84 and 82.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\python.exe' -m pytest imitation_learning/tests/test_feature_cache.py -q
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' imitation_learning/tests/test_setup_summary_integration.py
```

Expected: old widths/schema/signature assertions fail.

- [ ] **Step 3: Update cache and model constants**

Set cache schema to 18 and summary widths to 84/82 in feature cache and
network. Change all three feature-signature producers to
`numeric-summary-27-known-deck-setup-v4`. Do not change sparse token count or
decoder layouts.

- [ ] **Step 4: Run focused tests and commit**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\python.exe' -m pytest imitation_learning/tests/test_feature_cache.py -q
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' imitation_learning/tests/test_setup_summary_integration.py
git add imitation_learning/training/feature_cache.py imitation_learning/training/cache_features.py imitation_learning/model/network.py imitation_learning/training/train.py imitation_learning/validation/evaluate.py imitation_learning/tests/test_feature_cache.py
git add -f imitation_learning/tests/test_setup_summary_integration.py
git commit -m "feat: consume expanded setup summaries"
```

### Task 4: Kaggle Submission Feature Parity

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Create: `imitation_learning/tests/test_submission_setup_summary.py`

**Interfaces:**
- Consumes: checkpoint summary projections with 84/82 input widths
- Produces: generated `main.py` with the same setup catalog, evolution graph, Energy deficit, and asymmetric summaries

- [ ] **Step 1: Write a failing generated-source test**

Extract the `%%writefile main.py` cell. Assert constants 84/82, required helper
names, use of `Pokemon.energies`, no opponent-hand call, deck argument flow,
and successful `compile(source, 'main.py', 'exec')`.

- [ ] **Step 2: Run the test and verify RED**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\python.exe' -m pytest imitation_learning/tests/test_submission_setup_summary.py -q
```

Expected: old summary-width assertion fails.

- [ ] **Step 3: Patch only the generated-agent cell**

Mirror Tasks 1 and 2 in the notebook. Preserve model manifest handling,
ensemble probability averaging, deck selection, known-deck tracking, action
history, and policy selection.

- [ ] **Step 4: Run parity tests and commit**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\python.exe' -m pytest imitation_learning/tests/test_submission_known_deck.py imitation_learning/tests/test_submission_setup_summary.py -q
& 'C:\Users\Liuluotu\Anaconda\anaconda3\python.exe' -m json.tool imitation_learning/kaggle_submission_imitation_agent.ipynb > $null
git add imitation_learning/kaggle_submission_imitation_agent.ipynb
git add -f imitation_learning/tests/test_submission_setup_summary.py
git commit -m "feat: add setup summaries to Kaggle agent"
```

### Task 5: Final Regression Verification

**Files:**
- Verify only; no planned production edits

**Interfaces:**
- Confirms: extractor compatibility, cache rejection, model construction, and submission syntax

- [ ] **Step 1: Run all feature-specific tests**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\python.exe' -m pytest imitation_learning/tests/test_known_deck_tracker.py imitation_learning/tests/test_feature_cache.py imitation_learning/tests/test_submission_known_deck.py imitation_learning/tests/test_submission_setup_summary.py -q
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' imitation_learning/tests/test_setup_feature_catalog.py
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' imitation_learning/tests/test_player_setup_summary.py
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' imitation_learning/tests/test_setup_summary_integration.py
```

- [ ] **Step 2: Compile modified Python and validate artifacts**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\python.exe' -m py_compile imitation_learning/model/features.py imitation_learning/model/network.py imitation_learning/training/cache_features.py imitation_learning/training/feature_cache.py imitation_learning/training/train.py imitation_learning/validation/evaluate.py
git diff --check
git status --short
```

- [ ] **Step 3: Report migration requirements**

State explicitly: no replay extraction, mandatory feature re-cache, mandatory
retraining, and new checkpoints only for Kaggle inference.
