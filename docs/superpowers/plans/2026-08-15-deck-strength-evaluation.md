# Greedy Deck Strength Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, multiprocessing evaluator that uses one trained policy with Greedy Top-1 decisions to compare every configured pair of different Decks and output a directed win-rate matrix and ranking.

**Architecture:** A dedicated `deck_strength` package owns its config, schedule, Greedy battle loop, persistent workers, and reports. It reuses only the existing production `PolicyAgent` loader/feature path and CG engine API. The parent schedules unordered Deck pairs and writes reports; spawn-based workers independently load the same checkpoint and run games.

**Tech Stack:** Python 3.10+, PyTorch, CG engine API, `concurrent.futures`, PyYAML, NumPy, Matplotlib, pytest.

## Global Constraints

- Do not modify training, model architecture, feature cache, Beam Search behavior, or Kaggle submission behavior.
- Every unordered pair of different Decks gets exactly `games_per_pair` attempted games total.
- Alternate player seats; odd counts may differ by one game.
- Both players use Greedy Top-1 from the same checkpoint with independent histories.
- Use spawn-based persistent workers and keep at most `workers * 2` jobs outstanding.
- Exclude draws and failures from decisive-game win-rate denominators.
- Do not include diagonal mirror games.

---

### Task 1: Configuration and Unordered Matchup Schedule

**Files:**
- Create: `imitation_learning/deck_strength/__init__.py`
- Create: `imitation_learning/deck_strength/config.py`
- Create: `imitation_learning/tests/test_deck_strength_config.py`

**Interfaces:**
- Produces: `DeckSpec(name: str, cards: tuple[int, ...])`.
- Produces: `RuntimeSettings(workers: int, torch_threads_per_worker: int)`.
- Produces: `DeckStrengthSettings(cg_path, checkpoint, device, seed, games_per_pair, output, runtime, decks)`.
- Produces: `GameSpec(game_id, deck_a, deck_b, deck_a_player, seed)`.
- Produces: `load_settings(path: Path | None = None) -> DeckStrengthSettings`.
- Produces: `schedule_games(settings) -> tuple[GameSpec, ...]`.

- [ ] **Step 1: Write failing config and scheduling tests**

Create a temporary two-Deck YAML and assert strict values/path resolution. For
three Decks and `games_per_pair=3`, assert nine games, exactly the unordered
pairs `(a,b)`, `(a,c)`, `(b,c)`, no diagonal pair, three games per pair, seat
counts differing by at most one, sequential IDs, and unique seeds. Add invalid
cases for zero games/workers/threads, duplicate names, and non-60-card Decks.

- [ ] **Step 2: Run the config tests and verify import failure**

```powershell
python -m pytest imitation_learning/tests/test_deck_strength_config.py -q
```

Expected: fail because `deck_strength.config` does not exist.

- [ ] **Step 3: Implement strict dataclasses, YAML parsing, and scheduling**

Use `itertools.combinations(settings.decks, 2)` and assign:

```python
deck_a_player = pair_game_index % 2
seed = settings.seed + game_id
```

Validate all required mappings and values with explicit `ValueError` messages.

- [ ] **Step 4: Run config tests and verify success**

```powershell
python -m pytest imitation_learning/tests/test_deck_strength_config.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add imitation_learning/deck_strength imitation_learning/tests/test_deck_strength_config.py
git commit -m "feat: configure deck strength matchups"
```

### Task 2: Two-Sided Greedy Battle Loop

**Files:**
- Create: `imitation_learning/deck_strength/battle.py`
- Create: `imitation_learning/tests/test_deck_strength_battle.py`

**Interfaces:**
- Produces: `GameResult` with game identity, Deck names/seats, winner, both Deck outcomes, duration, selections, and error.
- Produces: `CgBattleBackend.start/select/finish/observation`.
- Produces: `run_game(game, policy, backend=None) -> GameResult`.
- Consumes: `PolicyHistory` and the policy's `greedy(observation, deck, history, record=True)`.

- [ ] **Step 1: Write failing battle tests**

Use a fake backend with one selection and a terminal observation. Assert the
acting player's Deck is passed to `policy.greedy`, player histories are
different objects, Deck A's result is correct regardless of seat, selections
are counted, and `finish()` is called. Add a start/select exception test that
returns `outcome_a=outcome_b="failed"` and preserves the error text.

- [ ] **Step 2: Run battle tests and verify import failure**

```powershell
python -m pytest imitation_learning/tests/test_deck_strength_battle.py -q
```

- [ ] **Step 3: Implement the minimal Greedy battle loop**

Map seats using:

```python
decks = {
    game.deck_a_player: game.deck_a,
    1 - game.deck_a_player: game.deck_b,
}
histories = {0: PolicyHistory(), 1: PolicyHistory()}
```

At each nonterminal observation call Greedy Top-1 for `yourIndex` with
`record=True`. Derive `outcome_a` and `outcome_b` from the engine winner, and
always finish a successfully started battle in `finally`.

- [ ] **Step 4: Run battle tests and verify success**

```powershell
python -m pytest imitation_learning/tests/test_deck_strength_battle.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add imitation_learning/deck_strength/battle.py imitation_learning/tests/test_deck_strength_battle.py
git commit -m "feat: run greedy deck strength battles"
```

### Task 3: Persistent Multiprocessing Workers

**Files:**
- Create: `imitation_learning/deck_strength/runner.py`
- Create: `imitation_learning/tests/test_deck_strength_runner.py`

**Interfaces:**
- Produces: `WorkerRuntime(policy, battle_backend).run(game) -> GameResult`.
- Produces: `build_worker_runtime`, `initialize_worker`, `run_worker_game`, `clear_worker_runtime`.
- Produces: `collect_bounded_results(executor, games, worker_function, max_pending, on_result)`.
- Produces: `execute_games(settings, games, on_result=None) -> list[GameResult]` sorted by game ID.

- [ ] **Step 1: Write failing runtime tests**

Assert worker initialization builds one reusable runtime, uninitialized worker
calls raise clearly, sequential execution returns every result and calls the
progress callback, and bounded collection returns results sorted by game ID.
Use a `ThreadPoolExecutor` only for the lightweight bounded-collection test.

- [ ] **Step 2: Run runner tests and verify import failure**

```powershell
python -m pytest imitation_learning/tests/test_deck_strength_runner.py -q
```

- [ ] **Step 3: Implement persistent sequential and spawn paths**

`build_worker_runtime` calls `torch.set_num_threads`, then
`beam_search.model_agent.load_policy_agent(settings)`, and creates one
`CgBattleBackend`. Parallel mode uses:

```python
ProcessPoolExecutor(
    max_workers=settings.runtime.workers,
    mp_context=multiprocessing.get_context("spawn"),
    initializer=initialize_worker,
    initargs=(settings,),
)
```

Let broken-pool and initializer failures propagate; ordinary failed games are
returned normally.

- [ ] **Step 4: Run runner tests and verify success**

```powershell
python -m pytest imitation_learning/tests/test_deck_strength_runner.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add imitation_learning/deck_strength/runner.py imitation_learning/tests/test_deck_strength_runner.py
git commit -m "feat: parallelize deck strength games"
```

### Task 4: Directed Matrix, Ranking, and Artifacts

**Files:**
- Create: `imitation_learning/deck_strength/reporting.py`
- Create: `imitation_learning/tests/test_deck_strength_reporting.py`

**Interfaces:**
- Produces: `AggregateReport(matchups, matrix, completed, ranking, overall)`.
- Produces: `aggregate_results(results, deck_names) -> AggregateReport`.
- Produces: `write_outputs(output, results, report, deck_names) -> None`.

- [ ] **Step 1: Write failing reporting tests**

Build results containing A wins, B wins, a draw, and a failure. Assert `[A,B]`
uses A outcomes, `[B,A]` uses B outcomes, diagonal values are `None`, draws and
failures are excluded from decisive win rate, ranking is descending, and these
six files are produced: `games.csv`, `matchup_summary.csv`,
`win_rate_matrix.csv`, `win_rate_matrix.png`, `deck_ranking.csv`, and
`summary.json`.

- [ ] **Step 2: Run reporting tests and verify import failure**

```powershell
python -m pytest imitation_learning/tests/test_deck_strength_reporting.py -q
```

- [ ] **Step 3: Implement aggregation and output writers**

Use one helper that summarizes a sequence of perspective outcomes. Construct
both directed matrix cells from the same canonical pair rows. Sort ranking by:

```python
(-win_rate_if_present, -wins, deck_name)
```

Render with a fixed `0..1` color scale, Deck rows, opponent columns, empty gray
diagonal, and `rate + n` annotations.

- [ ] **Step 4: Run reporting tests and verify success**

```powershell
python -m pytest imitation_learning/tests/test_deck_strength_reporting.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add imitation_learning/deck_strength/reporting.py imitation_learning/tests/test_deck_strength_reporting.py
git commit -m "feat: report greedy deck strength matrix"
```

### Task 5: Entrypoint, Default YAML, Documentation, and Real Smoke

**Files:**
- Create: `imitation_learning/deck_strength/evaluate.py`
- Create: `imitation_learning/cfg/deck_strength.yaml`
- Modify: `imitation_learning/README.md`
- Modify: `imitation_learning/tests/test_deck_strength_runner.py`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: runnable `python deck_strength/evaluate.py` workflow.

- [ ] **Step 1: Write a failing progress formatter test**

Assert the formatter contains completed/total, game ID, Deck pair, seats,
outcome, seconds, selections, and error when present.

- [ ] **Step 2: Run the focused test and verify failure**

```powershell
python -m pytest imitation_learning/tests/test_deck_strength_runner.py -q
```

- [ ] **Step 3: Implement the evaluator entrypoint**

Load config, schedule games, print run/worker/device totals, warn about multiple
CUDA model copies, stream completion lines, aggregate sorted results, write all
outputs, and print each Deck's final ranking row.

- [ ] **Step 4: Add a runnable default configuration and README section**

Use `final_d2_33.pt` as a project-relative editable checkpoint example, include
the currently configured project Deck lists, default to 100 games per pair and
two workers, and document output semantics and the run command.

- [ ] **Step 5: Run focused regression tests and compilation**

```powershell
python -m pytest imitation_learning/tests/test_deck_strength_config.py imitation_learning/tests/test_deck_strength_battle.py imitation_learning/tests/test_deck_strength_runner.py imitation_learning/tests/test_deck_strength_reporting.py imitation_learning/tests/test_beam_search_model_agent.py -q
python -m compileall -q imitation_learning/deck_strength
```

- [ ] **Step 6: Run a real two-worker CUDA smoke**

Use two configured Decks and two games total in a temporary smoke entrypoint.
Assert both game IDs return without error, generate all six outputs in a
temporary directory, then delete the smoke entrypoint and temporary outputs.

- [ ] **Step 7: Commit**

```powershell
git add imitation_learning/deck_strength/evaluate.py imitation_learning/cfg/deck_strength.yaml imitation_learning/README.md imitation_learning/tests/test_deck_strength_runner.py
git commit -m "feat: add greedy deck strength evaluator"
```
