# Festival Lead Beam Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix hidden-card determinization and replace the full Beam matrix with a Festival Lead Beam-versus-Greedy baseline comparison against every other configured Deck.

**Architecture:** Correct Deck conservation at the source by accounting for serial-deduplicated `select.effect` and `state.looking` cards. Refocus the existing `beam_search` configuration, schedule, battle result, reporting, and entrypoint around one target Deck and two conditions (`beam`, `greedy`) while retaining the persistent spawn worker implementation.

**Tech Stack:** Python 3.10+, PyTorch, CG engine/Search API, PyYAML, NumPy, Matplotlib, pytest.

## Global Constraints

- `contextCard` must not alter Deck conservation.
- Setup (`turn == 0`) must never call determinization or Beam Search.
- For each non-target opponent, run exactly `games_per_matchup` games for each of `beam` and `greedy`.
- Alternate the target Deck's player seat independently inside each condition.
- Greedy baseline must never invoke the CG Search API.
- All outcome metrics are from the target Deck's perspective.
- Preserve checkpoint, Deck lists, runtime workers, and search settings in the user's YAML.

---

### Task 1: Correct Transient Visible-Card Accounting

**Files:**
- Modify: `imitation_learning/beam_search/determinization.py`
- Modify: `imitation_learning/tests/test_beam_search_determinization.py`

**Interfaces:**
- Preserve: `build_search_inputs(obs, your_deck, opponent_deck, rng, basic_card_ids=...)`.
- Add internal collection of normal-zone Card serials and transient effect/looking cards.

- [ ] **Step 1: Write failing tests**

Give all fake cards unique `serial` values. Add tests where an own `effect` card
accounts for a one-card pool mismatch, seven `looking` cards account for a
seven-card mismatch, an effect already present in discard is not removed twice,
and `contextCard` does not change a previously valid pool.

- [ ] **Step 2: Run and confirm the new tests fail for hidden-zone mismatch**

```powershell
python -m pytest imitation_learning/tests/test_beam_search_determinization.py -q
```

- [ ] **Step 3: Implement serial-deduplicated transient accounting**

Collect normal-zone Card-like objects, create a `seen_serials` set, and append
`select.effect` plus `state.looking` Card IDs only when `playerIndex` matches and
the serial is new. Do not inspect `contextCard` or `select.deck` for removal.

- [ ] **Step 4: Verify focused tests and replay invariant**

Run the test file, then scan `7.9.jsonl.gz` non-setup states and assert all
7,205 own-player hidden-pool deltas are zero.

- [ ] **Step 5: Commit**

```powershell
git add imitation_learning/beam_search/determinization.py imitation_learning/tests/test_beam_search_determinization.py
git commit -m "fix: account for transient cards in Beam search"
```

### Task 2: Festival-Only Two-Condition Schedule

**Files:**
- Modify: `imitation_learning/beam_search/config.py`
- Modify: `imitation_learning/tests/test_beam_search_config.py`
- Modify: `imitation_learning/tests/test_beam_search_runner.py`

**Interfaces:**
- Add: `BeamSearchSettings.target_deck: str`.
- Replace `GameSpec` fields with `condition`, `target_deck`, `opponent_deck`, `target_player`, and existing ID/seed.
- Preserve: `schedule_games(settings) -> tuple[GameSpec, ...]`.

- [ ] **Step 1: Write failing schedule tests**

For three Decks, target `b`, and three games, assert 12 scheduled games:
two opponents × two conditions × three games. Assert no `b` mirror, conditions
are exactly `beam/greedy`, seats are balanced per opponent/condition, and an
unknown target name is rejected.

- [ ] **Step 2: Run config tests and confirm failure**

```powershell
python -m pytest imitation_learning/tests/test_beam_search_config.py -q
```

- [ ] **Step 3: Implement target validation and schedule**

Resolve the target by exact configured name. For every other Deck and each
condition, generate `games_per_matchup` entries with `target_player=index % 2`.
Assign sequential IDs and `seed + game_id`.

- [ ] **Step 4: Update runner fixtures to the new GameSpec and verify**

```powershell
python -m pytest imitation_learning/tests/test_beam_search_config.py imitation_learning/tests/test_beam_search_runner.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add imitation_learning/beam_search/config.py imitation_learning/tests/test_beam_search_config.py imitation_learning/tests/test_beam_search_runner.py
git commit -m "feat: schedule Festival Beam comparisons"
```

### Task 3: Beam and Greedy Target Battle Conditions

**Files:**
- Modify: `imitation_learning/beam_search/battle.py`
- Modify: `imitation_learning/tests/test_beam_search_reporting.py`

**Interfaces:**
- Replace `GameResult` identity with `condition`, `target_deck`, `opponent_deck`, and `target_player`.
- Preserve existing search diagnostics.
- Preserve: `run_game(...) -> GameResult`.

- [ ] **Step 1: Write failing behavior tests**

Test that `beam` invokes determinization/search only when the target acts after
setup; `greedy` invokes neither; target setup calls policy Greedy directly; the
opponent always uses Greedy; and target outcomes are correct in either seat.

- [ ] **Step 2: Run battle tests and confirm failure**

```powershell
python -m pytest imitation_learning/tests/test_beam_search_reporting.py -q
```

- [ ] **Step 3: Implement condition routing and setup bypass**

Use Beam only when:

```python
game.condition == "beam" and actor == game.target_player and obs.current.turn > 0
```

All other decisions use Greedy with `record=True`. Record search diagnostics
only for Beam decisions and interpret the winner relative to `target_player`.

- [ ] **Step 4: Verify behavior and search regressions**

```powershell
python -m pytest imitation_learning/tests/test_beam_search_reporting.py imitation_learning/tests/test_beam_search_search.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add imitation_learning/beam_search/battle.py imitation_learning/tests/test_beam_search_reporting.py
git commit -m "feat: compare Beam and Greedy target conditions"
```

### Task 4: Festival Comparison Reporting and Entrypoint

**Files:**
- Modify: `imitation_learning/beam_search/plot.py`
- Modify: `imitation_learning/beam_search/evaluate.py`
- Modify: `imitation_learning/cfg/beam_search.yaml`
- Modify: `imitation_learning/README.md`
- Modify: `imitation_learning/tests/test_beam_search_reporting.py`
- Modify: `imitation_learning/tests/test_beam_search_runner.py`

**Interfaces:**
- Produce report fields: `comparisons`, `overall_beam`, `overall_greedy`, `overall_uplift`.
- Produce files: `games.csv`, `opponent_summary.csv`, `festival_comparison.csv`, `festival_win_rate_comparison.png`, `summary.json`.

- [ ] **Step 1: Write failing aggregation/output/progress tests**

Construct Beam and Greedy results for two opponents. Assert per-opponent and
overall decisive win rates, `beam_uplift`, draw/failure exclusion, all five
artifacts, and progress lines containing condition/target/opponent/seat.

- [ ] **Step 2: Run reporting and runner tests and confirm failure**

```powershell
python -m pytest imitation_learning/tests/test_beam_search_reporting.py imitation_learning/tests/test_beam_search_runner.py -q
```

- [ ] **Step 3: Implement focused aggregation and grouped-bar output**

Summarize target outcomes by condition and opponent. Write the same comparison
rows to both requested CSV names. Plot Beam and Greedy bars on a shared 0-to-1
axis and annotate completed counts.

- [ ] **Step 4: Refocus evaluate.py and minimally update YAML/README**

Add only `target_deck: festival_lead` to the user's YAML and preserve every
other user edit. Print total opponents, conditions, games, progress, overall
Beam/Greedy rates, uplift, fallback rate, and output path.

- [ ] **Step 5: Run focused regression and compile checks**

```powershell
python -m pytest imitation_learning/tests/test_beam_search_config.py imitation_learning/tests/test_beam_search_determinization.py imitation_learning/tests/test_beam_search_search.py imitation_learning/tests/test_beam_search_reporting.py imitation_learning/tests/test_beam_search_runner.py imitation_learning/tests/test_beam_search_model_agent.py -q
python -m compileall -q imitation_learning/beam_search
```

- [ ] **Step 6: Run real two-worker CUDA smoke**

Temporarily select Festival plus one opponent with two games per condition.
Assert four games, both conditions, no hidden-zone conservation warning, and all
five outputs. Delete temporary smoke files afterward.

- [ ] **Step 7: Commit without capturing unrelated YAML edits**

Stage code/tests/docs normally and stage only the `target_deck` YAML hunk in
addition to the already tracked runtime block.

```powershell
git commit -m "feat: focus Beam evaluation on Festival Lead"
```
