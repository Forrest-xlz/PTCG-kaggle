# Policy-Only Beam Search Evaluation Design

## Goal

Add an isolated local evaluation package that measures whether policy-only
beam search improves game win rate over greedy policy inference. Both agents
use the same compact inference checkpoint. The only experimental difference
is action selection: one agent searches candidate trajectories while the other
selects the current policy Top-1 action.

This work must not change the training pipeline, feature cache, model
architecture, or Kaggle submission notebook.

## Package Layout

```text
imitation_learning/
|-- beam_search/
|   |-- __init__.py
|   |-- config.py
|   |-- model_agent.py
|   |-- search.py
|   |-- battle.py
|   |-- evaluate.py
|   `-- plot.py
`-- cfg/
    `-- beam_search.yaml
```

The package uses the correct `beam_search` spelling and remains independent of
the existing training and validation entry points.

## Configuration

`cfg/beam_search.yaml` contains one top-level `beam_search` mapping:

```yaml
beam_search:
  cg_path: ../pokemon_tcg_ai_battle/sample_submission
  checkpoint: outputs/model.inference-fp16.pt
  device: cuda
  seed: 42
  games_per_matchup: 100
  output: outputs/beam_search

  search:
    beam_width: 4
    expansion_top_k: 4
    alpha: 1.0
    max_depth: 64

  decks:
    - name: grimmsnarl
      cards: []
    - name: alakazam
      cards: []
```

Every configured deck must have a unique non-empty name and exactly 60
non-negative integer Card IDs. `games_per_matchup`, `beam_width`,
`expansion_top_k`, and `max_depth` must be positive integers. `alpha` must be a
finite non-negative number. The checkpoint must contain `model` and `config`
mappings and is the only source of model architecture parameters.

The empty card lists above are illustrative only. The checked-in default YAML
will contain valid project deck lists so it can be edited and run directly.

## Policy Agent

`model_agent.py` loads one inference checkpoint, creates the existing
`PTCGTransformer`, loads weights strictly, places it on the configured device,
and calls `eval()`.

It reuses the production feature path:

- `encoder_features`
- `decoder_features`
- `enumerate_actions`
- the checkpoint's saved `ModelConfig`

It enumerates at most 64 legal combination actions, matching training and the
Kaggle agent. For each state it returns the actions, logits, softmax
probabilities, and log probabilities. Probabilities are computed in FP32 and
clamped to a small positive minimum before applying `log`.

Each real agent owns an independent three-action history. Every hypothetical
beam branch copies its own history and appends its simulated actions, so one
branch cannot contaminate another. A simulated opponent decision uses a
separate opponent history. It does not reuse the beam player's history.

## Hidden-State Determinization

The CG Search API requires concrete identities for hidden cards. At every real
beam decision, the evaluator creates one reproducible determinization:

1. Start from each player's configured 60-card multiset.
2. Remove cards currently visible for that player, including visible hand,
   Active, Bench, attached Tool and Energy cards, discard, visible prizes, and
   other attributable public cards.
3. Shuffle the remaining multiset with a seed derived from the global seed,
   game ID, and decision ID.
4. Slice the shuffled multiset to fill the required hidden hand, prize, and
   remaining-deck counts.

Observed cards always remain unchanged. All branches under a root search share
the same sampled hidden state. A visible card that cannot be removed from the
configured deck is recorded as a warning, which accommodates generated or
transformed cards without creating negative counts. An impossible final set of
zone sizes makes the search fail safely and use greedy fallback.

This first version deliberately uses one determinization per real decision.
Multi-determinization aggregation is outside the scope of this baseline.

## Beam Search

Beam search uses the CG `search_begin`, `search_step`, `search_release`, and
`search_end` APIs to obtain real game transitions.

For a root controlled by player `p`, a trajectory stores:

- current CG Search State ID and observation;
- first root action;
- accumulated log policy probability for player `p`;
- number of scored decisions made by player `p`;
- separate simulated action histories;
- depth and completion/truncation status.

At a node controlled by player `p`, the policy evaluates every enumerated
legal action and the node expands its `expansion_top_k` highest-probability
actions. Each child adds its selected action's log probability and increments
the scored decision count.

At a node controlled by the opponent, the same checkpoint selects one greedy
Top-1 action. That action advances the state but contributes neither log
probability nor a step to the beam score.

After expansion, at most `beam_width` unfinished trajectories are retained.
Finished trajectories move into a separate completed pool and remain eligible
even while other branches continue. Search normally ends for a trajectory when
either:

- `state.turn` differs from the root turn, meaning the root player's turn has
  ended; or
- the game has ended.

Pre-game states with `turn == 0` bypass beam search and use greedy inference.
`max_depth` is only a safety bound. A trajectory that reaches it is treated as
completed for scoring and counted as truncated.

For a trajectory with `T` scored beam-player decisions, the final and pruning
score is:

```text
score = sum(log(policy_probability_t)) / (T ** alpha)
```

Only trajectories with `T >= 1` are eligible. Once no unfinished branches
remain, the best trajectory from the completed pool returns its first action
to the real battle. If no valid completed trajectory exists, the agent returns
the real root policy's greedy Top-1 action.

## Evaluation Schedule

For `D` configured decks, the evaluator runs every ordered matrix cell:

```text
Beam(deck_row) versus Greedy(deck_column)
```

This includes diagonal mirror cells. Each cell contains exactly
`games_per_matchup` attempted games. Beam and Greedy alternate player seats;
for an odd count, seat totals differ by at most one. Because both `(A, B)` and
`(B, A)` cells are run, each deck is evaluated both as the searching deck and
as the greedy opponent.

The evaluator is single-process in the baseline. This avoids unsafe sharing of
CG global pointers and duplicate CUDA model allocations. Parallel engine
workers and batched inference are separate performance work.

## Outputs

The configured output directory contains:

```text
games.csv
matchup_summary.csv
beam_win_rate_matrix.csv
beam_win_rate_matrix.png
summary.json
```

`games.csv` records one row per attempted game, including deck names, Beam
seat, result, elapsed time, selection counts, search calls, expanded nodes,
maximum depth, truncations, greedy fallbacks, warnings, and any error.

`matchup_summary.csv` contains wins, losses, draws, failures, completed games,
Beam win rate excluding draws, Beam non-loss rate, mean duration, and aggregate
search diagnostics for every ordered cell.

`beam_win_rate_matrix.csv` stores the numeric Beam win rate matrix. The PNG has
Beam decks on rows and Greedy decks on columns. Each cell displays Beam win
rate and completed game count. `summary.json` contains the overall result,
per-Beam-deck result, per-Greedy-opponent result, runtime, and search diagnostic
totals.

Failed games are written to `games.csv` and excluded from win-rate
denominators. Draws are reported separately and excluded from the primary
win-rate denominator.

## Resource and Error Safety

Every root search uses `try/finally` to call `search_end`. Search states that
are pruned or abandoned are explicitly released. A branch-level CG error drops
only that branch. A root-level search or determinization error records a
fallback reason and uses greedy inference. A battle-level engine error records
the game failure, calls `battle_finish` when applicable, and continues with the
next scheduled game.

## Verification

Automated tests cover:

- YAML validation and path resolution;
- checkpoint architecture loading;
- trajectory length normalization;
- beam pruning and first-action propagation on a fake search graph;
- opponent actions not changing the score or `T`;
- root-turn termination and `max_depth` truncation;
- independent hypothetical histories;
- hidden-zone counts and deck multiset conservation;
- ordered matchup construction and seat balance;
- matrix aggregation, including failures and draws.

A small CG smoke run verifies that Greedy and Beam agents can complete games,
Search API state is released, and all output files are produced. No training,
cache regeneration, or replay extraction is required.
