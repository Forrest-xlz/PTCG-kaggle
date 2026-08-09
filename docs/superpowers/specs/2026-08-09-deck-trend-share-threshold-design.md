# Deck-Trend Share Threshold Design

## Goal

Give `analysis.min_share_percent` one consistent meaning based on the relevant
population: daily share for time-series and Sankey display, and pooled
game-weighted share for the matchup matrix.

## Semantics

- Daily line charts: an archetype/date point is visible only when that day's
  `share_percent >= min_share_percent`. Both usage and win-rate values are
  hidden below the threshold, and the line is broken across hidden dates.
- Sankey: an archetype below the threshold on a given date is mapped to
  `Other` for that date. Existing behavior is retained.
- Matchup matrix: an archetype is included only when its pooled share across
  all selected snapshots meets the threshold. Pooled share is
  `sum(archetype uses) / sum(all uses) * 100`, not an unweighted mean of daily
  percentages.
- Full `archetype_daily_metrics.csv`, raw team modal archetypes, retention,
  inflow, outflow, and net flow remain unfiltered.

## Implementation

Add pure helpers in `deck/trend.py` for validating the threshold, creating the
daily visibility mask, and calculating pooled archetype shares. The notebook
uses a full date index plus NaN values to force Matplotlib to break lines at
hidden dates. The matchup cell derives its row/column set from pooled shares.

## Verification

Tests cover a deck that is above threshold on one date but below it on another,
and a deck whose pooled weighted share falls below threshold despite a high
single-day share. Notebook JSON/source checks confirm that peak-share global
selection is removed.
