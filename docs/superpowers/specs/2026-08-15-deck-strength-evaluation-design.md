# Greedy Deck Strength Evaluation Design

## Goal

Add an isolated evaluator that compares the strength of configured Decks when
the same trained policy controls both sides with Greedy Top-1 action selection.
For every pair of different Decks, it runs a configured total number of games,
alternates player seats, and produces per-game results, matchup summaries, a
win-rate matrix, a matrix image, and an overall Deck ranking.

This feature does not change training, model architecture, feature extraction,
feature caching, Beam Search behavior, or Kaggle submission behavior.

## Package Layout

```text
imitation_learning/
|-- deck_strength/
|   |-- __init__.py
|   |-- config.py
|   |-- battle.py
|   |-- runner.py
|   |-- reporting.py
|   `-- evaluate.py
`-- cfg/
    `-- deck_strength.yaml
```

The evaluator may reuse stable policy-loading and feature-encoding components
from `beam_search.model_agent`, but it does not import Beam Search execution or
scoring logic.

## Configuration

`cfg/deck_strength.yaml` contains one `deck_strength` mapping:

```yaml
deck_strength:
  cg_path: ../pokemon_tcg_ai_battle/sample_submission
  checkpoint: outputs/model.inference-fp16.pt
  device: cuda
  seed: 42
  games_per_pair: 100
  output: outputs/deck_strength

  runtime:
    workers: 2
    torch_threads_per_worker: 1

  decks:
    - name: deck_a
      cards: []
    - name: deck_b
      cards: []
```

Every Deck must have a unique non-empty name and exactly 60 non-negative Card
IDs. `games_per_pair`, `workers`, and `torch_threads_per_worker` must be
positive integers. Paths are resolved relative to the `imitation_learning`
project directory, matching the existing Beam Search evaluator.

The checked-in configuration will contain valid project Deck lists rather than
the illustrative empty lists above.

## Matchup Schedule

For `D` Decks, the evaluator schedules each unordered pair exactly once, for a
total of:

```text
D * (D - 1) / 2 * games_per_pair
```

There are no diagonal same-Deck mirror games. For each pair `(A, B)`, exactly
`games_per_pair` games are attempted. Player seats alternate by game index. If
the count is odd, the seat counts differ by one at most.

Each game has a deterministic `game_id` and seed derived from the configured
base seed. Scheduling order and multiprocessing completion order do not change
these assignments.

## Greedy Policy Battle

Both players use the same loaded checkpoint and the existing production policy
feature path. At every selection, the acting player's legal combination actions
are enumerated and the action with maximum policy probability is selected.

Each player owns an independent three-action policy history. Actions by player
0 cannot modify player 1's history. No Beam Search, hidden-state
determinization, rollout, value head, sampling, or temperature is used.

A worker loads one policy model and reuses it for both players and for every
game assigned to that worker. The configured Deck determines the engine Deck
and the encoder Deck input; the policy parameters remain identical across all
Decks.

## Multiprocessing

The evaluator uses the same process isolation principles as the Beam Search
evaluator:

- the parent constructs the deterministic schedule and writes reports;
- every worker is started with the `spawn` context;
- every worker independently loads the checkpoint onto the configured device;
- every worker owns independent CG state and a process-local battle backend;
- only settings, game specifications, and result records cross processes;
- at most `workers * 2` games remain outstanding;
- returned results are sorted by `game_id` before reporting.

With `workers: 1`, evaluation runs sequentially. With one GPU and multiple
workers, GPU and host memory use increase because each worker owns a model copy.
Two workers is the default baseline.

## Results and Matrix Semantics

For a matchup between Deck A and Deck B, the reports record outcomes from each
Deck's perspective. The matrix uses:

- rows: evaluated Deck;
- columns: opponent Deck;
- cell `[A, B]`: A's wins divided by A's wins plus A's losses against B;
- cell `[B, A]`: B's corresponding win rate;
- diagonal cells: empty;
- draws: reported separately and excluded from the primary win-rate denominator;
- failed games: reported separately and excluded from all win-rate denominators.

For decisive games, opposing cells are complementary. Draws can make non-loss
rates non-complementary, so the primary matrix remains decisive-game win rate.

## Outputs

The configured output directory contains:

```text
games.csv
matchup_summary.csv
win_rate_matrix.csv
win_rate_matrix.png
deck_ranking.csv
summary.json
```

`games.csv` contains one row per attempted game: pair, seats, winner, duration,
selection count, outcome from both Deck perspectives, and any error.

`matchup_summary.csv` contains one canonical row per unordered Deck pair with
attempted/completed games, wins for each Deck, losses, draws, failures,
decisive-game win rates, non-loss rates, and mean duration.

`win_rate_matrix.csv` and `win_rate_matrix.png` contain the directed matrix
described above. Every plotted cell shows win rate and completed-game count.

`deck_ranking.csv` aggregates all non-mirror opponents for each Deck and sorts
by decisive-game win rate descending, then wins descending, then Deck name.
It includes attempted games, completed games, wins, losses, draws, failures,
win rate, non-loss rate, and mean game duration.

`summary.json` contains the overall run totals, matchup rows, and ranking rows.

## Failure Handling

A battle-level engine or inference exception creates a failed game result. The
worker calls `battle_finish` when a battle was successfully started, then
continues with later games. A normal failed game does not terminate evaluation.

A worker initialization failure, process crash, or broken process pool aborts
the run. The parent does not write a partial report that could be mistaken for
a complete comparison.

## Verification

Automated tests cover:

- strict YAML validation and path resolution;
- unordered-pair scheduling and seat balance;
- deterministic game IDs and seeds;
- both players using Greedy Top-1 with independent histories;
- cleanup after successful and failed battles;
- sequential and bounded spawn-based scheduling;
- matrix direction, complementary decisive outcomes, draws, and failures;
- ranking order and all required output files.

A real two-worker CUDA smoke run uses two Decks and two games. It verifies that
both worker model copies load, both seats complete, CG states remain isolated,
and all reports can be generated. A larger performance run is left to the user
because its duration depends on the configured Deck count and games per pair.
