# Pure-Python Deck Trends EDA Design

## Purpose

Replace the Deck Trends analysis notebook with one configuration-driven Python
EDA script that produces only the three useful figures and two supporting audit
tables. Separate replay-trend extraction settings from visualization settings,
and give each figure an independent date, share, mirror, and score-selection
policy.

## Scope

This change will:

- split deck-trend extraction and EDA into separate YAML files;
- replace `notebooks/deck_trends.ipynb` with
  `notebooks/deck_trends_eda.py`;
- generate one combined share/win-rate line figure, one Sankey figure, and one
  matchup matrix per run;
- give each figure one independently configured parameter set;
- support `all`, `min`, `max`, and `avg` replay-score filtering;
- retain the reusable calculations and plotting helpers in
  `analysis/deck_trends.py`;
- write figures and audit tables beneath one configured output directory;
- update tests and README instructions for the new workflow.

This change will not:

- convert any other EDA notebook;
- add command-line configuration flags;
- generate multiple variants of the same figure in one run;
- change deck classification rules;
- change replay-trend extraction schemas or existing cached trend data;
- add training features or modify model behavior.

## Target Files

```text
imitation_learning/
|-- analysis/
|   `-- deck_trends.py
|-- cfg/
|   |-- deck_trends_eda.yaml
|   `-- extract_deck_trend_data.yaml
|-- extraction/
|   `-- deck_trend_data.py
`-- notebooks/
    `-- deck_trends_eda.py
```

`notebooks/deck_trends.ipynb` and the combined
`cfg/analyze_deck_trends.yaml` will no longer exist. The `notebooks/` directory
remains the project's home for human-facing EDA artifacts even when a focused
EDA is a Python script rather than an interactive notebook.

## Commands

Run both commands from `imitation_learning/`:

```bash
python -m extraction.deck_trend_data
python notebooks/deck_trends_eda.py
```

Both commands continue to use fixed default YAML paths. No `--config` option
will be introduced.

## Extraction Configuration

`cfg/extract_deck_trend_data.yaml` contains only extraction settings:

```yaml
extract:
  input: ../replay_episodes
  output: data/deck_trend
  workers: 16
  limit_members: null
  force: false
```

`extraction/deck_trend_data.py` will read this file. Existing archive
fingerprinting, incremental reuse, output schema, and force-rebuild behavior
remain unchanged.

## EDA Configuration

`cfg/deck_trends_eda.yaml` contains only analysis inputs, output location, and
the three figure specifications:

```yaml
analysis:
  input: data/deck_trend
  replay_episodes: ../replay_episodes
  card_table: ../pokemon_tcg_ai_battle/EN_Card_Data.csv
  output: outputs/deck_trends

  line_chart:
    start_date: 8.12
    end_date: 8.15
    interval_days: 1
    min_share_percent: 1.0
    exclude_mirror_matches: false
    score_mode: all
    score_threshold: null
    filename: archetype_share_and_win_rate.png

  sankey:
    start_date: 8.12
    end_date: 8.15
    interval_days: 1
    min_share_percent: 1.0
    score_mode: min
    score_threshold: 1060
    filename: archetype_sankey.png

  matchup_matrix:
    start_date: 8.12
    end_date: 8.15
    interval_days: 1
    min_share_percent: 1.0
    exclude_mirror_matches: true
    score_mode: min
    score_threshold: 1000
    filename: archetype_matchup_matrix.png
```

Paths are resolved relative to `imitation_learning/`. Dates use the existing
`month.day` representation and are inclusive. Each `interval_days` value must
be a positive integer, and each `min_share_percent` must be in `[0, 100]`.

`filename` must be a plain filename ending in `.png`; it may not escape the
configured output directory. The three filenames must be distinct.

## Score Selection

Score selection operates at replay level. When a replay passes, both player
rows remain in the analysis.

The supported modes are:

| Mode | Included replay condition |
| --- | --- |
| `all` | Every replay in the selected dates; `score_threshold` must be `null`. |
| `min` | `min_score >= score_threshold`. Both participants have reached the threshold. |
| `max` | `(sum_score - min_score) >= score_threshold`. At least one participant has reached the threshold. |
| `avg` | `avg_score >= score_threshold`. The two-participant mean has reached the threshold. |

For `min`, `max`, and `avg`, `score_threshold` must be a finite number. The EDA
script reads `manifest.csv` from each selected replay ZIP, validates a unique
row per `(date, episode_id)`, and joins scores to both player rows using those
keys. Missing or inconsistent score data is an error.

`score_mode: all` does not require loading manifests for that figure. Manifest
data should be loaded once and shared by all score-filtered figures in the run.

## Figure Semantics

### Share and Win-Rate Line Figure

The line figure contains two vertically stacked panels:

1. daily archetype share;
2. daily archetype win rate.

The figure independently selects dates, applies its score filter, and computes
daily metrics from the remaining replays. An archetype point is visible on a
date only when its share on that date is at least `min_share_percent`.

`exclude_mirror_matches` affects only the win-rate numerator and denominator.
Usage and share always include every retained player row. The exported metric
column should be named `win_rate`, with metadata or logging stating whether
mirror matches were excluded; the misleading unconditional
`non_mirror_win_rate` presentation will not be retained in the EDA output.

### Sankey Figure

The Sankey figure shows how each team's modal archetype changes between
adjacent selected snapshots. Its score filter is applied before modal
archetypes and flows are calculated.

For each date independently, archetypes below `min_share_percent` are grouped
into `Other`. The Sankey does not expose `exclude_mirror_matches`, because team
archetype flow does not calculate battle win rate and mirror exclusion would
have no effect.

### Matchup Matrix

The matchup figure pools all of its selected dates after applying its score
filter. Archetype eligibility is calculated from pooled usage inside this same
filtered dataset. Both matrix rows and columns must have pooled share at least
`min_share_percent`.

`exclude_mirror_matches: true` removes same-archetype matchups and leaves the
matrix diagonal empty. `false` includes the diagonal and its games in the long
table. Each populated cell displays win rate and game count, with rows meaning
"row archetype against column archetype."

## Output

One run creates:

```text
outputs/deck_trends/
|-- archetype_share_and_win_rate.png
|-- archetype_sankey.png
|-- archetype_matchup_matrix.png
`-- tables/
    |-- line_chart_daily_metrics.csv
    `-- matchup_matrix.csv
```

Actual PNG names come from the YAML. Table filenames remain stable so they are
easy to find.

`line_chart_daily_metrics.csv` contains the complete daily metric rows behind
the line figure, including rows hidden by the share threshold. Its columns
include date, archetype, uses, share percentage, wins, losses, draws, evaluated
games, and win rate.

`matchup_matrix.csv` is the long-form table behind the matrix. Its columns are
row archetype, column archetype, games, wins, losses, draws, and win rate. It
contains the matchup data after the matrix figure's date, score, and mirror
filters; share eligibility remains visible through which archetypes are
included.

The Sankey has no table export. `selected_dates.csv` will not be generated.
The script logs each figure's selected dates, score filter, retained replay
count, share threshold, output path, and mirror policy when applicable.

## Code Organization

`notebooks/deck_trends_eda.py` owns configuration loading, input loading,
figure-specific orchestration, rendering, and output writes. It must expose a
`main()` function and guard execution with:

```python
if __name__ == "__main__":
    main()
```

Reusable transformations remain in `analysis/deck_trends.py`. New reusable
helpers will cover:

- validated chart settings;
- replay score reconstruction and filtering;
- figure-specific row/date selection;
- the combined line plot;
- matchup heatmap rendering.

The existing Sankey layout/rendering helpers remain reusable. The orchestration
script must not duplicate deck classification, daily metric, team-flow, or
matchup aggregation algorithms.

Matplotlib figures are explicitly closed after saving so repeated runs do not
retain GUI state. The script is non-interactive and does not call
`IPython.display.display` or `plt.show()`.

## Failure Behavior

The EDA command exits with a clear exception when:

- a required YAML section or field is absent;
- a type, date range, threshold, mode, or filename is invalid;
- no extracted trend shards exist;
- a selected archive or its `manifest.csv` is missing when scores are needed;
- replay/player pairs or score joins are inconsistent;
- a figure retains no dates, replays, archetypes, flows, or matchups;
- configured PNG filenames collide.

Outputs are written only after the corresponding figure data has validated.
Existing unrelated outputs are not deleted. A failed figure must not silently
produce an empty or misleading image.

## Testing and Verification

Tests will cover:

- extraction and EDA configurations load independently;
- each score mode uses the specified inclusive `>=` condition;
- `all` requires a null threshold and score modes require a finite threshold;
- line visibility is calculated per date;
- Sankey visibility is calculated per date after its own score filter;
- matrix visibility is calculated from pooled usage after its own score filter;
- mirror exclusion affects line win rate and matchups, not usage share;
- one controlled fixture run writes exactly three PNGs and two CSV tables;
- the line CSV contains hidden low-share rows;
- the old notebook and combined YAML no longer exist;
- README commands and paths match the new workflow;
- `git diff --check` passes.

The existing three baseline failures in deck-analysis tests are outside this
change. They must be distinguished from new Deck Trends EDA regressions.

## Migration

1. Move the extraction section unchanged to
   `cfg/extract_deck_trend_data.yaml`.
2. Create `cfg/deck_trends_eda.yaml` with the three independent figure specs.
3. Update `extraction/deck_trend_data.py` to use the extraction-only YAML.
4. Add reusable score/filter/rendering helpers with focused tests.
5. Port the three useful notebook figures into
   `notebooks/deck_trends_eda.py`.
6. Verify equivalent calculations on controlled fixtures.
7. Remove `notebooks/deck_trends.ipynb`.
8. Update README and obsolete-path tests/searches.

The current working copy of `notebooks/deck_trends.ipynb` contains executed
outputs. The approved pure-Python migration intentionally removes that
notebook; no other local changes or generated data will be deleted.
