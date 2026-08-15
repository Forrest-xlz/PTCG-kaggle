# Beam Search Multiprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run independent Beam-versus-Greedy games concurrently with persistent process-local model and CG state on one configured GPU.

**Architecture:** Add validated runtime settings, isolate construction of process-local evaluation objects in a new runner module, and let the parent schedule a bounded number of games through a spawn-based `ProcessPoolExecutor`. Sequential mode shares the same worker runtime path, while the parent remains solely responsible for progress, deterministic result ordering, aggregation, and output files.

**Tech Stack:** Python 3.10+, PyTorch, `concurrent.futures`, `multiprocessing`, PyYAML, pytest.

## Global Constraints

- Use the multiprocessing `spawn` context on every platform.
- Every worker owns an independent model, CUDA context, CG backend, and search backend.
- Never pass models, CG handles, observations, or search states between processes.
- Preserve matchup scheduling, per-game seeds, policy/search behavior, reporting, and all user-edited checkpoint/deck YAML values.
- Default to `workers: 2` and `torch_threads_per_worker: 1`.
- `workers: 1` must retain a sequential debugging path.

---

### Task 1: Runtime Configuration

**Files:**
- Modify: `imitation_learning/beam_search/config.py`
- Modify: `imitation_learning/tests/test_beam_search_config.py`

**Interfaces:**
- Produces: `RuntimeSettings(workers: int, torch_threads_per_worker: int)`.
- Produces: `BeamSearchSettings.runtime: RuntimeSettings`.
- Consumes: `beam_search.runtime.workers` and `beam_search.runtime.torch_threads_per_worker` from YAML.

- [ ] **Step 1: Write failing configuration tests**

Extend `_write_config` with:

```python
"runtime": {"workers": 2, "torch_threads_per_worker": 1},
```

Add assertions:

```python
assert settings.runtime.workers == 2
assert settings.runtime.torch_threads_per_worker == 1
```

Add invalid-value cases:

```python
(("runtime", "workers"), 0, "runtime.workers"),
(("runtime", "torch_threads_per_worker"), False, "torch_threads_per_worker"),
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
python -m pytest imitation_learning/tests/test_beam_search_config.py -q
```

Expected: failure because `BeamSearchSettings` has no `runtime` member.

- [ ] **Step 3: Implement strict runtime parsing**

Add:

```python
@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    workers: int
    torch_threads_per_worker: int
```

Add `runtime: RuntimeSettings` to `BeamSearchSettings`, require the `runtime`
mapping, and parse both values with `_positive_int`.

- [ ] **Step 4: Run the focused tests and verify success**

Run:

```powershell
python -m pytest imitation_learning/tests/test_beam_search_config.py -q
```

Expected: all configuration and scheduling tests pass.

- [ ] **Step 5: Commit the configuration change**

```powershell
git add imitation_learning/beam_search/config.py imitation_learning/tests/test_beam_search_config.py
git commit -m "feat: configure beam search workers"
```

### Task 2: Persistent Worker Runtime and Bounded Scheduling

**Files:**
- Create: `imitation_learning/beam_search/runner.py`
- Create: `imitation_learning/tests/test_beam_search_runner.py`

**Interfaces:**
- Produces: `WorkerRuntime(policy, searcher, battle_backend, basic_card_ids)`.
- Produces: `build_worker_runtime(settings: BeamSearchSettings) -> WorkerRuntime`.
- Produces: `initialize_worker(settings: BeamSearchSettings) -> None`.
- Produces: `run_worker_game(game: GameSpec) -> GameResult`.
- Produces: `execute_games(settings, games, on_result=None) -> list[GameResult]`.
- Consumes: existing `load_policy_agent`, `BeamSearcher`, `CgSearchBackend`, `CgBattleBackend`, and `run_game`.

- [ ] **Step 1: Write failing worker and scheduling tests**

Create tests that monkeypatch `build_worker_runtime` with a fake runtime and
assert initialization is reused:

```python
def test_worker_initialization_is_reused(monkeypatch, settings, games):
    built = []
    runtime = FakeRuntime()
    monkeypatch.setattr(runner, "build_worker_runtime", lambda value: built.append(value) or runtime)
    runner.initialize_worker(settings)
    first = runner.run_worker_game(games[0])
    second = runner.run_worker_game(games[1])
    assert len(built) == 1
    assert [first.game_id, second.game_id] == [0, 1]
```

Test sequential dispatch and completion callback:

```python
def test_execute_games_sequential_reports_every_result(monkeypatch, settings, games):
    monkeypatch.setattr(runner, "build_worker_runtime", lambda value: FakeRuntime())
    seen = []
    actual = runner.execute_games(settings, games, on_result=seen.append)
    assert [item.game_id for item in actual] == [0, 1, 2]
    assert seen == actual
```

Test parallel completion ordering with an injectable executor factory that
returns real `Future` objects completed in reverse order. Assert that callbacks
observe completion order while the returned list is sorted by `game_id`.

- [ ] **Step 2: Run runner tests and verify failure**

Run:

```powershell
python -m pytest imitation_learning/tests/test_beam_search_runner.py -q
```

Expected: import failure because `beam_search.runner` does not exist.

- [ ] **Step 3: Implement process-local runtime construction**

Create:

```python
@dataclass(slots=True)
class WorkerRuntime:
    policy: Any
    searcher: Any
    battle_backend: Any
    basic_card_ids: frozenset[int]

    def run(self, game: GameSpec) -> GameResult:
        return run_game(
            game,
            self.policy,
            self.searcher,
            self.basic_card_ids,
            backend=self.battle_backend,
        )
```

`build_worker_runtime` calls `torch.set_num_threads`, loads the policy once,
reads `all_card_data`, and creates process-local search/battle backends.
`initialize_worker` stores the runtime in a module-private global.
`run_worker_game` raises a clear `RuntimeError` if initialization did not run.

- [ ] **Step 4: Implement sequential and bounded parallel execution**

For `workers == 1`, build one local runtime and run each game directly.
For multiple workers, create:

```python
context = multiprocessing.get_context("spawn")
ProcessPoolExecutor(
    max_workers=settings.runtime.workers,
    mp_context=context,
    initializer=initialize_worker,
    initargs=(settings,),
)
```

Keep at most `workers * 2` futures outstanding. Use `wait(...,
return_when=FIRST_COMPLETED)`, invoke `on_result` as each result arrives, refill
the pending set, then return all results sorted by `game_id`. Let initializer,
pickling, and broken-pool exceptions propagate.

- [ ] **Step 5: Run runner tests and verify success**

Run:

```powershell
python -m pytest imitation_learning/tests/test_beam_search_runner.py -q
```

Expected: worker lifecycle, sequential scheduling, bounded parallel scheduling,
and deterministic returned ordering tests pass.

- [ ] **Step 6: Commit the runner**

```powershell
git add imitation_learning/beam_search/runner.py imitation_learning/tests/test_beam_search_runner.py
git commit -m "feat: run beam games in persistent workers"
```

### Task 3: Evaluator Integration and User Configuration

**Files:**
- Modify: `imitation_learning/beam_search/evaluate.py`
- Modify: `imitation_learning/cfg/beam_search.yaml`
- Modify: `imitation_learning/README.md`
- Modify: `imitation_learning/tests/test_beam_search_runner.py`

**Interfaces:**
- Consumes: `execute_games(settings, games, on_result)` from Task 2.
- Preserves: existing `aggregate_results`, `write_outputs`, progress fields,
  and overall summary output.

- [ ] **Step 1: Write a failing progress-format test**

Extract a pure formatter and test it independently:

```python
def test_format_progress_contains_completion_and_game_identity():
    line = format_progress(3, 12, result(8))
    assert "[3/12]" in line
    assert "game_id=8" in line
    assert "beam=a" in line
    assert "greedy=b" in line
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
python -m pytest imitation_learning/tests/test_beam_search_runner.py -q
```

Expected: failure because `format_progress` is not defined.

- [ ] **Step 3: Replace inline evaluation with the runner**

Move progress-line construction to:

```python
def format_progress(completed: int, total: int, result: GameResult) -> str:
    ...
```

In `main`, print worker/device information, emit a CUDA duplication warning
when `workers > 1` and the device starts with `cuda`, call `execute_games`, and
retain existing aggregation and output logic. Do not load a model or CG backend
in the parent when parallel mode is active.

- [ ] **Step 4: Add runtime defaults without disturbing user YAML values**

Insert only:

```yaml
  runtime:
    workers: 2                    # Independent model + CG worker processes.
    torch_threads_per_worker: 1   # Avoid CPU thread oversubscription.
```

Do not rewrite the checkpoint path, deck lists, search settings, or output path.
Add a README note describing duplicated GPU memory and recommending comparison
of 1, 2, and optionally 4 workers.

- [ ] **Step 5: Run focused and regression tests**

Run:

```powershell
python -m pytest imitation_learning/tests/test_beam_search_config.py imitation_learning/tests/test_beam_search_runner.py imitation_learning/tests/test_beam_search_search.py imitation_learning/tests/test_beam_search_reporting.py imitation_learning/tests/test_beam_search_determinization.py imitation_learning/tests/test_beam_search_model_agent.py -q
python -m compileall -q imitation_learning/beam_search
```

Expected: all focused Beam Search tests pass and compilation succeeds.

- [ ] **Step 6: Run a minimal real smoke evaluation**

Temporarily use one small matchup/game with `workers: 1`, then with `workers: 2`.
Confirm both runs return every scheduled game and write complete artifacts. Do
not treat throughput from one game as a benchmark. If the configured checkpoint
or CG library is unavailable in the current environment, report the exact
environmental blocker and rely on the automated tests.

- [ ] **Step 7: Commit implementation files without capturing unrelated YAML edits**

Stage code, tests, and README normally. Stage only the added `runtime` YAML hunk
so the user's checkpoint and deck edits remain their own working-tree changes.

```powershell
git add imitation_learning/beam_search/evaluate.py imitation_learning/README.md imitation_learning/tests/test_beam_search_runner.py
git commit -m "feat: parallelize beam search evaluation"
```
