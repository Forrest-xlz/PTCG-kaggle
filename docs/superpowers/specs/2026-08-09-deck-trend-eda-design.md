# Deck Trend EDA Design

## Goal

Add a reusable Deck trend analysis workflow that follows teams and archetypes
across dated replay snapshots. The baseline classifier uses only the existing
curated `ARCHETYPE_RULES`; decks that do not match a curated rule are grouped
as `Other`.

The workflow must keep expensive replay extraction separate from inexpensive
classification and analysis. Changing the snapshot interval or archetype
classifier must not require replay extraction to run again as long as the new
classifier depends only on the complete 60-card deck list.

## Scope

The change adds:

- a reusable replay-to-player-row extraction script;
- a YAML configuration file shared by extraction and analysis;
- a trend-specific archetype classifier that does not alter the current Deck
  EDA classifier;
- a reader-facing Jupyter notebook for trend, team-switching, and matchup
  analysis;
- compact cached extraction output and derived analysis tables under `data/`.

The change does not modify training data, validation splits, model code,
feature caches, the existing `deck_eda.ipynb`, or the behavior of
`deck.analysis.classify_deck()`.

## File Layout

```text
imitation_learning/
├── cfg/
│   └── deck_trend.yaml
├── deck/
│   ├── trend.py
│   └── trend_extract.py
├── eda/
│   └── deck_trend.ipynb
└── data/
    └── deck_trend/
        ├── <month.day>.players.csv.gz
        ├── <month.day>.meta.json
        └── analysis/
```

Per-date shards make extraction resumable and allow a new replay archive to be
processed without rewriting completed dates. The notebook concatenates these
shards at load time.

## Configuration

`cfg/deck_trend.yaml` contains two sections:

```yaml
extract:
  input: ../replay_episodes
  output: data/deck_trend
  workers: 16
  limit_members: null
  force: false

analysis:
  snapshot_interval_days: 3
  start_date: null
  end_date: null
  min_share_percent: 2.0
  exclude_mirror_matches: true
```

`snapshot_interval_days` must be a positive integer. A value of `1` selects
every available daily snapshot. A value of `3` starts at the earliest selected
date and then chooses the earliest available date at least three calendar days
after the preceding snapshot. This handles missing daily archives without
silently converting the parameter into an index stride.

`start_date` and `end_date` are optional inclusive `month.day` bounds.
`min_share_percent` controls chart visibility and Sankey node inclusion; it
does not delete rows from matchup or exported census data.

## Extraction

`deck/trend_extract.py` reads every matching replay ZIP in `extract.input`.
Archives are processed in parallel. Each valid replay becomes two player rows
with these columns:

| Column | Meaning |
|---|---|
| `date` | Date inferred from the replay archive filename |
| `episode_id` | Stable episode identifier |
| `player` | Player index, `0` or `1` |
| `team_name` | `info.TeamNames[player]`, with the agent name as fallback |
| `opponent_team_name` | Other player's team name |
| `deck` | Sorted JSON list containing all 60 Card IDs |
| `reward` | Final reward for this player |
| `result` | `win`, `loss`, or `draw` derived from the reward |

The extractor validates that each replay yields two team names and two valid
60-card decks. Invalid replays are skipped and summarized in the per-date meta
file with bounded error details.

When `force` is false, a completed shard with the current schema version is
reused. Newly added archives are processed independently. `limit_members` is
available only for smoke runs and is recorded in metadata so a limited shard
cannot be mistaken for a complete extraction.

The output deliberately retains complete Deck Card IDs and does not persist an
archetype label. This is the contract that allows classifiers to change without
re-extracting replay archives.

## Baseline Archetype Classifier

`deck/trend.py` exposes a trend-specific classifier. It reuses the existing
ordered `ARCHETYPE_RULES` and card-name normalization but does not invoke any
fallback classification:

```text
first matching ARCHETYPE_RULES entry -> curated archetype name
no matching entry                    -> Other
```

Rule ordering remains significant. More specific rules such as
`Great Tusk / Crustle` must precede the generic `Crustle Wall` rule.

The classifier is separate from `classify_deck()`. Existing Deck EDA and
validation-isolation outputs therefore remain stable. A later classifier may
replace or extend this baseline inside the trend workflow without changing the
extracted player-row schema.

## Notebook Data Flow

`eda/deck_trend.ipynb` is an analysis notebook with this top-to-bottom flow:

1. Load `cfg/deck_trend.yaml` and discover extracted date shards.
2. Validate the two-row replay invariant, dates, Deck lengths, results, and
   team names.
3. Select calendar snapshots using `snapshot_interval_days` and optional date
   bounds.
4. Deduplicate exact Deck lists, classify each unique Deck once, and map the
   result back to player rows.
5. Report curated-rule coverage and the `Other` share for every snapshot.
6. Compute archetype usage, non-mirror win rate, team modal archetype, adjacent
   snapshot flows, retention, switching, and matchup tables.
7. Render bounded charts and export reusable derived tables.

The notebook reads cached extracted data only. It never scans replay ZIP files.

## Analyses and Visuals

### Classification Coverage

Show total games, player rows, unique teams, selected dates, curated-rule
coverage, and `Other` share. A high `Other` share is reported as a limitation of
the baseline rather than silently hidden.

### Usage and Win-Rate Trends

For each selected snapshot, compute archetype uses and share of player rows.
Compute win rate against the field with same-archetype mirror matches excluded
when configured. Show bounded trend tables and line charts for archetypes whose
share reaches `min_share_percent` in at least one selected snapshot.

### Team Modal Archetype

A team's archetype for a snapshot is the most frequently used archetype among
that team's player rows on that date. Ties are resolved deterministically by
archetype name so reruns are stable.

### Sankey Flow

For every pair of adjacent selected snapshots, include teams appearing in both
dates and count transitions between their modal archetypes. Nodes represent
snapshot archetypes; ribbons represent both retained archetypes and switches.
Low-share nodes are grouped into `Other` for display only.

### Retention and Switching

For each adjacent snapshot pair, report unique teams, overlapping teams,
retention from each side, unchanged modal archetypes, switched modal
archetypes, and switch rate among overlapping teams. Also report archetype
inflow and outflow counts.

### Matchup Matrix

Pool games from selected snapshots and compute row-archetype win rate against
column archetype together with the game count. Mirror cells are omitted when
`exclude_mirror_matches` is true. Counts remain visible so small samples are not
presented as reliable matchup conclusions.

## Exported Analysis Tables

The notebook writes derived CSV files under `data/deck_trend/analysis/`:

- `snapshot_dates.csv`;
- `archetype_daily_metrics.csv`;
- `team_modal_archetypes.csv`;
- `team_snapshot_flows.csv`;
- `team_retention.csv`;
- `matchup_matrix_long.csv`.

These outputs are derived and may be overwritten whenever configuration or the
classifier changes.

## Error Handling and Validation

Extraction fails early for an invalid YAML shape, missing input directory,
non-positive worker count, or invalid `force`/`limit_members` values. A broken
replay member does not terminate the entire archive; its error is recorded.

The notebook fails with an actionable message when no shards exist, no dates
remain after filtering, player rows are not paired by replay, or Deck JSON is
invalid. It checks that usage shares sum to 100 percent per selected date and
that matchup wins and losses reconcile with replay results.

Static tests cover date parsing and interval selection, baseline classification,
team modal tie-breaking, and transition counts. The notebook should be executed
top-to-bottom against a smoke extraction before handoff. If the local runtime
lacks Python/Jupyter, the exact missing runtime and execution command must be
reported.

## Re-extraction Contract

Changing any of the following requires only rerunning the notebook:

- snapshot interval or date bounds;
- share threshold or mirror handling;
- artificial archetype rules or their priority;
- replacement of the baseline with another Deck-Card-ID-only classifier.

Extraction is required again only when replay archives are added or replaced,
the extraction schema changes, or a future analysis needs replay information
that is not present in the cached player rows.
