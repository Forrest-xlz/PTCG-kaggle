# Beam Search Multiprocessing Design

## Goal

Parallelize independent Beam-versus-Greedy games on a machine with one GPU.
Each worker process owns its own CG engine state, beam-search objects, and CUDA
model copy. The matchup schedule, random seed assigned to each game, search
algorithm, scoring formula, and generated reports remain unchanged.

This design supersedes only the single-process evaluation statement in
`2026-08-15-beam-search-evaluation-design.md`.

## Chosen Approach

Use a spawn-based process pool with persistent worker initialization.

Each worker initializes exactly once:

1. load the YAML-derived evaluation settings;
2. load an independent policy model on the configured device;
3. construct its own `BeamSearcher`, `CgSearchBackend`, and `CgBattleBackend`;
4. load the immutable set of basic Card IDs.

The parent sends `GameSpec` values to the pool. A worker runs one full game and
returns the existing game-result record. It then reuses the same initialized
model and engine objects for subsequent games.

This is preferred over CPU-only inference because the trained model remains on
the GPU. It is preferred as the baseline over a centralized GPU inference
server because the latter would require asynchronous IPC throughout variable-
length search trees and would substantially complicate the current evaluator.

## Configuration

Add a `runtime` mapping inside `beam_search`:

```yaml
beam_search:
  runtime:
    workers: 2
    torch_threads_per_worker: 1
```

- `workers` is the number of independent processes and must be at least one.
  `1` preserves a sequential execution path for debugging and comparison.
- `torch_threads_per_worker` controls CPU threads used by PyTorch inside each
  worker and must be at least one. The default is `1` to prevent CPU thread
  oversubscription while CG simulations run concurrently.

All workers use the existing `device` value. With `device: cuda`, this means
each worker loads a separate model on the same GPU. The evaluator prints a
startup warning when multiple workers share one CUDA device because GPU memory
usage grows approximately with the number of workers.

The checked-in/user-edited checkpoint and deck configuration must be preserved.

## Process and Data Flow

The entry point remains `beam_search/evaluate.py`.

1. The parent loads and validates configuration and builds the deterministic
   ordered game schedule.
2. With one worker, the parent uses the existing sequential behavior.
3. With multiple workers, the parent creates a multiprocessing context using
   `spawn` on every platform and starts a process pool.
4. The worker initializer builds process-local runtime objects once.
5. Scheduled games are submitted with a small bounded number of outstanding
   jobs so the parent does not create an unnecessarily large future list.
6. Results are reported as games finish, while retaining `game_id` in every
   record.
7. Before aggregation and file output, the parent sorts results by `game_id`,
   making output deterministic regardless of completion order.

Only lightweight settings, game specifications, and result records cross the
process boundary. PyTorch models, CG handles, observations, and search states
are never shared or pickled between processes.

## Correctness and Failure Handling

Game seeds continue to come from the deterministic schedule, so process timing
does not alter the assigned seed. Parallel execution may change wall-clock
ordering but not matchup membership or seat assignment.

A normal game-level failure remains an ordinary failed game result and does not
stop the pool. If a worker initializer fails, or a worker process terminates
unexpectedly, the parent aborts evaluation with a clear error instead of
silently omitting scheduled games. Already returned results are not written as
a misleading complete report.

Worker processes are closed and joined through a context manager. Existing CG
cleanup inside battle and search code remains responsible for per-game native
resources.

## Performance Expectations

This architecture improves throughput when CPU-side CG simulation leaves the
GPU idle between short inference calls. It does not guarantee linear scaling:
workers contend for one GPU and maintain separate CUDA contexts. Two workers is
the recommended initial value; four workers is an optional experiment if GPU
memory is sufficient and utilization remains low.

Startup becomes slower because every worker loads a model. This cost is
amortized across all games handled by that worker. Peak GPU memory and model
host memory increase roughly in proportion to `workers`.

## Verification

Automated tests will verify:

- runtime configuration validation;
- worker initialization constructs process-local runtime exactly once;
- sequential mode still runs all scheduled games;
- parallel mode returns every scheduled game even when completion order differs;
- output aggregation receives results sorted by `game_id`;
- a returned game-level failure does not abort remaining games;
- an initializer or broken-process failure is surfaced rather than ignored.

A small real smoke run should compare `workers: 1` and `workers: 2` with the
same seed and schedule. It should confirm that both write complete reports and
measure games per second and peak GPU memory. Exact game outcomes are not
required to match across separate runs if the underlying CG implementation has
uncontrolled randomness, but each run must contain the identical scheduled
game IDs, decks, and seat assignments.
