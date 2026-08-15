# Score-Filtered Deck Trend Views

## Goal

Extend `imitation_learning/eda/deck_trend.ipynb` with two optional views that
analyze only replays whose two-player mean score is above a cell-local
threshold:

1. a score-filtered Sankey flow immediately below the existing Sankey flow;
2. a score-filtered matchup matrix immediately below the existing matchup
   matrix.

The existing unfiltered analyses and extraction format remain unchanged.

## Score Source and Join

Each replay archive contains `manifest.csv`, keyed by `episode_id`, with an
`avg_score` column. The first score-filtered cell reads only the small manifest
from each ZIP represented in `snapshot_dates`, adds the archive stem as `date`,
and joins the result to `annotated` on `(date, episode_id)`.

The notebook validates that:

- every selected replay has exactly one manifest score;
- `avg_score` is numeric and finite;
- filtering retains or removes whole replays, so both player rows remain
  together.

No replay JSON is read and no Deck trend extraction needs to be rerun.

## Score-Filtered Sankey Cell

The new cell appears immediately after the existing Sankey output and exposes:

```python
SANKEY_MIN_AVG_SCORE = 1100.0
```

It filters with the strict condition
`avg_score > SANKEY_MIN_AVG_SCORE`, then recomputes team modal archetypes,
daily shares, visible archetypes, team transitions, and the Sankey layout from
the filtered rows. It retains the notebook's existing date snapshots,
archetype classifier, daily share threshold, and presentation style.

If a team has no qualifying replay on a selected date, it does not contribute
a node or transition for that date. The cell prints the threshold and replay
counts before and after filtering. If no replay remains, it raises a clear
`ValueError` instead of drawing an empty chart.

The scored/annotated rows prepared here are retained as
`annotated_with_scores` for reuse by the later matchup cell.

## Score-Filtered Matchup Cell

The new cell appears immediately after the existing matchup matrix and exposes
an independent threshold:

```python
MATCHUP_MIN_AVG_SCORE = 1100.0
```

It filters `annotated_with_scores` with the same strict greater-than rule,
calls the existing `build_matchups()` helper, and preserves
`EXCLUDE_MIRROR_MATCHES`. Matrix archetypes use the existing pooled/global
visibility rule so the filtered and unfiltered charts have the same inclusion
semantics.

The heatmap keeps the existing row-beats-column orientation, percentage plus
game-count annotations, color scale, and layout. Its title includes the score
threshold. The cell prints replay counts before and after filtering and raises
a clear error when the filtered set is empty.

The filtered long-form table is written under the existing analysis directory
with a threshold-specific filename, for example:

```text
matchup_matrix_avg_score_gt_1100_long.csv
```

## Scope and Compatibility

- Do not modify `deck/trend_extract.py` or extracted shard schemas.
- Do not alter existing charts, exports, configuration values, or outputs.
- Keep both thresholds local to their respective notebook cells.
- Reuse the manifests loaded for the Sankey view rather than reopening them in
  the matchup view.
- Continue to honor `snapshot_dates`, `MIN_SHARE_PERCENT`, and
  `EXCLUDE_MIRROR_MATCHES`.

## Validation

Validate the notebook structure with `nbformat`, then execute it top-to-bottom
when the configured data and notebook environment permit. Confirm that:

- the manifest join has no missing selected episodes;
- both filtered datasets contain exactly two rows per replay;
- matchup wins, losses, and draws sum to games;
- displayed win rates remain within `[0, 1]`;
- both original and filtered views remain present in the notebook.
