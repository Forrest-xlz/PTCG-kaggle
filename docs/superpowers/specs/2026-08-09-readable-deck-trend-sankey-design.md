# Readable Deck-Trend Sankey Design

## Goal

Replace the unreadable auto-arranged Plotly Sankey in `eda/deck_trend.ipynb`
with a static Matplotlib figure matching the supplied reference: fixed date
columns, compact percentage labels, stable archetype colors, a bottom legend,
and ribbons that show team deck transitions.

## Scope

The change affects only Sankey construction and rendering. Existing extraction,
archetype classification, date selection, daily metrics, CSV outputs, line
charts, team-modal calculation, and matchup analysis remain unchanged.

## Data Semantics

- Each date column represents one selected snapshot.
- Node height is proportional to that date's player-row usage share.
- Archetypes below `analysis.min_share_percent` on a date are merged into
  `Other` before layout and flow construction.
- Ribbons connect modal archetypes for teams present in two adjacent snapshots.
- Ribbon widths use one global teams-to-height scale, so widths are comparable
  across dates. Unused node height is allowed because not every team reappears
  in the adjacent snapshot.
- Each date's displayed shares sum to 100 percent.

## Layout

- Dates occupy fixed, evenly spaced x coordinates and are annotated above the
  corresponding columns.
- Nodes within each date are ordered by descending usage share, with
  deterministic archetype-name tie breaking.
- Node heights fill a common vertical band after reserving small inter-node
  gaps. Node width is constant.
- Ribbons are allocated deterministically inside source and destination nodes
  using the opposite endpoint's vertical order to reduce crossings.
- Cubic Bezier polygons connect source and destination allocations.

## Labels and Colors

- Nodes and outgoing ribbons use a deterministic categorical color per
  archetype; `Other` is gray.
- Node labels contain only the usage percentage and use a small translucent
  white background when needed for contrast.
- Full archetype names appear once in a multi-column legend below the chart.
- Snapshot dates appear above the chart and total games appear below each date.
- Detailed values remain available in the existing CSV outputs rather than
  being repeated as long node labels.

## Output and QA

The notebook displays the figure and writes
`data/deck_trend/analysis/archetype_sankey.png`. Automated checks cover share
layout, global ribbon scaling, deterministic node ordering, and successful
figure construction. Visual QA checks label overlap, date alignment, legend
completeness, consistent colors, and readable ribbons at the configured date
count.
