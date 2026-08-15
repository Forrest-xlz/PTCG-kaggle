# Policy-Only Beam Search Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated evaluator that compares a compact BC checkpoint using turn-level policy-only beam search against greedy inference across a configurable deck matchup matrix.

**Architecture:** A strict YAML loader creates immutable settings, a shared policy adapter reuses the production feature/model code, and a dependency-injected beam core advances real CG Search States. A battle runner schedules every ordered deck pair and a reporting module writes per-game data, summaries, and a win-rate heatmap.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, PyYAML, pandas, matplotlib, CG Battle/Search API, pytest.

## Global Constraints

- Create only the isolated `imitation_learning/beam_search` package, its YAML, tests, and README documentation.
- Do not modify training behavior, feature-cache schemas, model architecture, or the Kaggle notebook.
- Load model architecture exclusively from the compact checkpoint's `config` mapping.
- Enumerate no more than 64 actions, matching production inference.
- Search until the root player's turn changes or the game ends; `max_depth` is a safety cutoff.
- Score only root-player decisions as `sum(log_probability) / T**alpha`; opponent Top-1 decisions do not affect the score or `T`.
- Run the baseline in one process.

---

### Task 1: Configuration and Matchup Scheduling

**Files:**
- Create: `imitation_learning/beam_search/__init__.py`
- Create: `imitation_learning/beam_search/config.py`
- Create: `imitation_learning/cfg/beam_search.yaml`
- Test: `imitation_learning/tests/test_beam_search_config.py`

**Interfaces:**
- Produces: `DeckSpec(name: str, cards: tuple[int, ...])`, `SearchSettings`, `BeamSearchSettings`, `load_settings(path: Path | None = None)`, and `schedule_games(settings) -> tuple[GameSpec, ...]`.

- [ ] **Step 1: Write failing configuration and schedule tests**

```python
def test_schedule_contains_every_ordered_cell_and_balances_seats(settings):
    games = schedule_games(settings)
    assert {(g.beam_deck.name, g.greedy_deck.name) for g in games} == {
        ("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")
    }
    cell = [g for g in games if g.beam_deck.name == "a" and g.greedy_deck.name == "b"]
    assert len(cell) == settings.games_per_matchup
    assert abs(sum(g.beam_player == 0 for g in cell) - sum(g.beam_player == 1 for g in cell)) <= 1
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python -m pytest imitation_learning/tests/test_beam_search_config.py -q`

- [ ] **Step 3: Implement strict dataclasses, project-relative paths, YAML validation, and deterministic ordered scheduling**

```python
@dataclass(frozen=True, slots=True)
class GameSpec:
    game_id: int
    beam_deck: DeckSpec
    greedy_deck: DeckSpec
    beam_player: int
    seed: int
```

- [ ] **Step 4: Run configuration tests**

Run: `python -m pytest imitation_learning/tests/test_beam_search_config.py -q`

- [ ] **Step 5: Commit Task 1**

```bash
git add imitation_learning/beam_search imitation_learning/cfg/beam_search.yaml imitation_learning/tests/test_beam_search_config.py
git commit -m "feat: add beam search evaluation config"
```

### Task 2: Reusable Policy Inference Adapter

**Files:**
- Create: `imitation_learning/beam_search/model_agent.py`
- Test: `imitation_learning/tests/test_beam_search_model_agent.py`

**Interfaces:**
- Consumes: `DeckSpec`, checkpoint `model` and `config`, existing feature functions and `PTCGTransformer`.
- Produces: `PolicyOutput(actions, probabilities, log_probabilities, encoded_options)`, `PolicyHistory`, `PolicyAgent.evaluate(obs, deck, history)`, `PolicyAgent.greedy(...)`, and `load_policy_agent(settings)`.

- [ ] **Step 1: Write failing tests with a fake model and observation**

```python
def test_policy_output_softmax_and_greedy(fake_agent, obs):
    output = fake_agent.evaluate(obs, DECK, PolicyHistory())
    assert torch.isclose(output.probabilities.sum(), torch.tensor(1.0))
    assert fake_agent.greedy(obs, DECK, PolicyHistory()) == output.actions[int(output.probabilities.argmax())]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest imitation_learning/tests/test_beam_search_model_agent.py -q`

- [ ] **Step 3: Implement checkpoint loading and production feature tensor conversion**

```python
@dataclass(slots=True)
class PolicyHistory:
    actions: deque[HistoryActionFeatures] = field(default_factory=lambda: deque(maxlen=HISTORY_STEPS))

    def clone(self) -> "PolicyHistory":
        return PolicyHistory(deque(self.actions, maxlen=HISTORY_STEPS))
```

The adapter must append history only when `record_action` is explicitly called, allowing beam branches to clone before mutation.

- [ ] **Step 4: Run adapter tests**

Run: `python -m pytest imitation_learning/tests/test_beam_search_model_agent.py -q`

- [ ] **Step 5: Commit Task 2**

```bash
git add imitation_learning/beam_search/model_agent.py imitation_learning/tests/test_beam_search_model_agent.py
git commit -m "feat: add reusable beam policy adapter"
```

### Task 3: Hidden-State Determinization

**Files:**
- Create: `imitation_learning/beam_search/determinization.py`
- Test: `imitation_learning/tests/test_beam_search_determinization.py`

**Interfaces:**
- Consumes: root Observation, root/other `DeckSpec`, deterministic `random.Random`.
- Produces: `SearchInputs(your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, warnings)` and `build_search_inputs(...)`.

- [ ] **Step 1: Write failing multiset and reproducibility tests**

```python
def test_hidden_assignments_preserve_counts_and_seed(obs):
    first = build_search_inputs(obs, OWN, OPP, random.Random(7))
    second = build_search_inputs(obs, OWN, OPP, random.Random(7))
    assert first == second
    assert len(first.your_deck) == obs.current.players[obs.current.yourIndex].deckCount
    assert len(first.opponent_hand) == obs.current.players[1 - obs.current.yourIndex].handCount
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest imitation_learning/tests/test_beam_search_determinization.py -q`

- [ ] **Step 3: Implement duplicate-aware visible-card removal and hidden-zone slicing**

Use `collections.Counter`, recursively collect main Pokémon, tools, Energy cards, discard, visible prizes, and the acting player's visible hand. Missing visible cards append warnings instead of decrementing below zero. Validate that remaining pool length exactly covers every requested hidden zone.

- [ ] **Step 4: Run determinization tests**

Run: `python -m pytest imitation_learning/tests/test_beam_search_determinization.py -q`

- [ ] **Step 5: Commit Task 3**

```bash
git add imitation_learning/beam_search/determinization.py imitation_learning/tests/test_beam_search_determinization.py
git commit -m "feat: add reproducible hidden state sampling"
```

### Task 4: Policy-Only Beam Core

**Files:**
- Create: `imitation_learning/beam_search/search.py`
- Test: `imitation_learning/tests/test_beam_search_search.py`

**Interfaces:**
- Consumes: `SearchSettings`, a `SearchBackend` protocol, `PolicyAgent`, root and opponent decks/histories.
- Produces: `trajectory_score(log_sum: float, steps: int, alpha: float)`, `SearchStats`, and `BeamSearcher.choose(obs, ...) -> SearchDecision`.

- [ ] **Step 1: Write failing score, opponent-step, turn-stop, pruning, and fallback tests using a fake graph backend**

```python
def test_opponent_transition_does_not_change_score(fake_search):
    decision = fake_search.choose(ROOT)
    assert decision.action == [1]
    assert decision.stats.scored_steps == 2
    assert decision.stats.opponent_steps == 1

def test_score_is_length_normalized():
    assert trajectory_score(-6.0, 3, 1.0) == -2.0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest imitation_learning/tests/test_beam_search_search.py -q`

- [ ] **Step 3: Implement backend protocol, trajectory cloning, completed pool, pruning, resource release, and greedy fallback**

```python
class SearchBackend(Protocol):
    def begin(self, obs, inputs: SearchInputs): ...
    def step(self, search_id: int, action: list[int]): ...
    def release(self, search_id: int) -> None: ...
    def end(self) -> None: ...
```

Every `choose` call must wrap the root lifecycle in `try/finally: backend.end()` and release pruned states exactly once.

- [ ] **Step 4: Run beam-core tests**

Run: `python -m pytest imitation_learning/tests/test_beam_search_search.py -q`

- [ ] **Step 5: Commit Task 4**

```bash
git add imitation_learning/beam_search/search.py imitation_learning/tests/test_beam_search_search.py
git commit -m "feat: implement turn-level policy beam search"
```

### Task 5: Battle Execution and Reports

**Files:**
- Create: `imitation_learning/beam_search/battle.py`
- Create: `imitation_learning/beam_search/plot.py`
- Create: `imitation_learning/beam_search/evaluate.py`
- Test: `imitation_learning/tests/test_beam_search_reporting.py`

**Interfaces:**
- Consumes: `GameSpec`, policy agent, BeamSearcher, CG battle functions.
- Produces: `GameResult`, `run_game`, `aggregate_results`, `write_outputs`, and executable `evaluate.main()`.

- [ ] **Step 1: Write failing aggregation and matrix tests**

```python
def test_matrix_uses_beam_rows_and_greedy_columns(results):
    summary, matrix = aggregate_results(results, ("a", "b"))
    assert matrix.loc["a", "b"] == 0.75
    assert summary.query("beam_deck == 'a' and greedy_deck == 'b'").iloc[0].games == 4
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest imitation_learning/tests/test_beam_search_reporting.py -q`

- [ ] **Step 3: Implement safe game lifecycle and stateful agent histories**

`run_game` returns the configured deck when `obs.select is None`, alternates real agents according to `obs.current.yourIndex`, calls Beam only for its assigned player, and guarantees `battle_finish` in `finally` after a successful start.

- [ ] **Step 4: Implement CSV, JSON, annotated heatmap, progress logs, and executable main**

The heatmap annotation format is `"{win_rate:.1%}\nn={completed}"`. Primary Beam win rate excludes draws and failed games. Store draws, failures, non-loss rate, duration, and search diagnostics separately.

- [ ] **Step 5: Run reporting tests**

Run: `python -m pytest imitation_learning/tests/test_beam_search_reporting.py -q`

- [ ] **Step 6: Commit Task 5**

```bash
git add imitation_learning/beam_search imitation_learning/tests/test_beam_search_reporting.py
git commit -m "feat: add beam matchup runner and reports"
```

### Task 6: Documentation and End-to-End Verification

**Files:**
- Modify: `imitation_learning/README.md`
- Test: all beam-search tests and a configured two-game smoke run.

**Interfaces:**
- Consumes: completed beam-search package.
- Produces: documented invocation and verified artifacts.

- [ ] **Step 1: Document configuration semantics and execution**

```bash
cd imitation_learning
python beam_search/evaluate.py
```

Document that no extraction/cache step is required, CUDA uses one process, rows are Beam decks, columns are Greedy decks, and the checkpoint must be inference-compatible.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest imitation_learning/tests/test_beam_search_*.py -q`
Expected: all focused tests pass.

- [ ] **Step 3: Compile the new package**

Run: `python -m compileall -q imitation_learning/beam_search`
Expected: exit code 0.

- [ ] **Step 4: Run a small CG smoke evaluation when a configured checkpoint exists**

Temporarily use two valid decks and `games_per_matchup: 1`; verify that `games.csv`, `matchup_summary.csv`, `beam_win_rate_matrix.csv`, `beam_win_rate_matrix.png`, and `summary.json` exist. If the checkpoint path is not present locally, report the skipped smoke run explicitly rather than fabricating success.

- [ ] **Step 5: Check scope and working tree**

Run: `git diff --check` and `git status --short`.
Expected: only planned beam-search files, YAML, tests, and README are changed.

- [ ] **Step 6: Commit documentation**

```bash
git add imitation_learning/README.md
git commit -m "docs: explain beam search evaluation"
```
