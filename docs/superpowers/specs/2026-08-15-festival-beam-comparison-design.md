# Festival Lead Beam Comparison Design

## Goal

Replace the full Beam-versus-Greedy Deck matrix with a focused experiment that
measures whether Beam Search improves one configured target Deck, initially
`festival_lead`. For every other configured Deck, compare:

1. Beam Festival Lead versus a Greedy opponent;
2. Greedy Festival Lead versus the same Greedy opponent.

Each condition plays `games_per_matchup` games and alternates Festival's player
seat. The evaluator reports Festival's win-rate uplift from Greedy to Beam.

The work also fixes the hidden-card determinization bug that currently causes
every evaluated game to contain at least one Beam-to-Greedy fallback.

## Confirmed Determinization Root Cause

The current hidden pool removes cards in hand, discard, visible prizes, Active,
Bench, attached cards, evolution cards, and Stadium. It misses cards that have
temporarily left their normal zone during effect resolution:

- `Observation.select.effect`: the card whose effect is being processed. It
  may already have left the hand but not yet entered discard.
- `State.looking`: cards temporarily removed from the Deck while being viewed.

Replay evidence from 7,205 non-setup observations in `7.9.jsonl.gz` showed:

- current accounting: 1,033 mismatches;
- adding only `effect`: unresolved `looking` mismatches;
- adding only `looking`: unresolved `effect` mismatches;
- adding both with serial-number deduplication: 7,205/7,205 exact matches.

`select.contextCard` is a contextual reference rather than evidence that a card
left its zone. Adding it caused over-removal and must not be used for pool
conservation.

At `turn == 0`, a player's selected Active can remain face-down, producing a
one-card mismatch that cannot be supplied through the Search API's input for
the acting player. Beam Search is already not useful during setup, so setup
decisions must bypass determinization and use Greedy directly.

## Configuration

Keep the existing `beam_search` YAML hierarchy and replace the full-matrix
meaning with a target-Deck experiment:

```yaml
beam_search:
  target_deck: festival_lead
  games_per_matchup: 100
```

`target_deck` must exactly match one configured Deck name. At least one other
Deck must remain. `games_per_matchup` applies independently to each condition
and each opponent.

With `D` configured Decks, total attempted games are:

```text
2 * (D - 1) * games_per_matchup
```

Existing checkpoint, CG path, device, runtime worker, search, output, and Deck
configuration remain valid.

## Schedule and Experimental Conditions

For each configured opponent other than the target Deck, schedule two
conditions:

- `beam`: Festival uses Beam; the opponent uses Greedy.
- `greedy`: Festival and the opponent both use Greedy.

Each condition has exactly `games_per_matchup` games. Festival alternates
between player 0 and player 1; odd counts differ by at most one seat. There are
no Festival mirror games and no opponent-versus-opponent games.

Every game stores a stable `game_id`, condition, target Deck, opponent Deck,
target player seat, and deterministic seed. Results are always interpreted
from Festival's perspective.

## Battle Execution

Both conditions load the same checkpoint and use the same feature/history
implementation.

For the `beam` condition:

- Festival uses turn-level Beam Search after setup;
- the opponent always selects Greedy Top-1;
- Festival setup decisions at `turn == 0` use Greedy Top-1 without attempting
  determinization.

For the `greedy` condition, both players always select Greedy Top-1 and no
determinization or CG Search API call occurs.

Each player owns an independent policy history. Festival and opponent histories
are not shared across players, games, conditions, or worker processes.

## Corrected Visible-Card Accounting

Determinization will collect visible Card objects and serial numbers for each
player from normal zones. It then considers `select.effect` and `state.looking`
cards whose `playerIndex` matches that player. A transient card is removed from
the remaining Deck multiset only if its unique `serial` has not already appeared
in a normal visible zone or earlier transient entry.

This serial deduplication prevents a referenced effect card already present in
discard, field, or another visible zone from consuming a second copy with the
same Card ID. Unknown/generated Card IDs retain the existing warning behavior.

The exact hidden-zone equality checks remain strict after corrected accounting.
Unexpected conservation failures still trigger Greedy fallback and remain
visible in diagnostics; the code must not silently trim or pad pools.

## Multiprocessing

The existing spawn-based persistent worker architecture remains unchanged.
Each worker owns one model, CG battle state, search backend, and basic-card
catalog. Games from both conditions may execute in any completion order, but
the parent sorts results by `game_id` before reporting.

## Outputs

The configured output directory contains:

```text
games.csv
opponent_summary.csv
festival_comparison.csv
festival_win_rate_comparison.png
summary.json
```

`games.csv` records condition, target/opponent Deck, Festival seat, outcome,
duration, selections, Beam search diagnostics, fallbacks, warnings, and error.

`opponent_summary.csv` and `festival_comparison.csv` contain one row per
opponent with separate Beam and Greedy attempted/completed games, wins, losses,
draws, failures, decisive-game win rates, non-loss rates, mean durations, and:

```text
beam_uplift = beam_win_rate - greedy_win_rate
```

Uplift is empty if either condition has no decisive games.

The PNG is a grouped bar chart by opponent with Beam and Greedy Festival win
rates on the same 0-to-1 scale. Each bar is annotated with completed-game count;
the title states that rates exclude draws and failed games.

`summary.json` contains overall Beam Festival metrics, overall Greedy Festival
metrics, overall uplift, per-opponent comparison rows, runtime, and aggregate
Beam search diagnostics.

## Failure Handling

Ordinary battle failures remain failed game rows and do not stop later games.
A Beam root determinization/search error produces a Greedy fallback and warning,
not a missing result. A worker initialization failure or broken process pool
aborts the run so a partial result set is not written as complete.

## Verification

Tests cover:

- effect cards removed exactly once by serial;
- looking cards removed exactly once by serial;
- context cards do not alter conservation;
- all sampled non-setup replay states conserve Deck size after the fix;
- setup decisions do not invoke determinization or Beam Search;
- Festival-only two-condition scheduling, counts, exclusions, and seat balance;
- Greedy condition never invokes Beam Search;
- target-perspective outcomes independent of Festival seat;
- per-opponent and overall Beam uplift calculations;
- all five output artifacts and chart labels;
- sequential and multiprocessing regressions.

A real two-worker CUDA smoke uses Festival plus one opponent with two games per
condition. It must finish all four games, write all artifacts, and show no
hidden-zone conservation fallback. Larger statistical runs remain user-driven.
