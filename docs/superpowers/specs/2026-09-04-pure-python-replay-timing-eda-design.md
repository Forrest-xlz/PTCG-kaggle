# Pure Python Replay Timing EDA Design

## Goal

Replace the mixed extraction-and-analysis `replay_timing.ipynb` workflow with two focused Python commands:

1. extract player timing records from replay archives;
2. analyze one extracted date and write reproducible charts and useful tables.

The change also standardizes EDA output naming: replay timing and deck trends use fixed filenames owned by their scripts instead of configurable `filename` fields.

## Scope

### In scope

- Add a replay timing extraction module and extraction YAML.
- Add a pure Python replay timing EDA script and EDA YAML.
- Preserve all six useful plots currently produced by the notebook.
- Export the three agreed analysis tables.
- Support `date: latest` and an explicit archive date such as `8.15` in both commands.
- Replace the old replay timing notebook.
- Remove chart `filename` fields from the deck trends EDA YAML and use fixed names in Python.
- Make both EDA workflows replace their owned outputs only after a complete successful run.
- Update documentation and tests affected by the new paths and interfaces.

### Out of scope

- Changing replay timing definitions or score formulas.
- Adding new plots or style controls.
- Converting other EDA notebooks.
- Changing the deck trends calculations or chart contents.
- Making extraction output replacement transactional; the atomic replacement requirement applies to EDA outputs.

## File Layout

```text
imitation_learning/
|-- cfg/
|   |-- extract_replay_timing.yaml
|   |-- replay_timing_eda.yaml
|   `-- deck_trends_eda.yaml
|-- extraction/
|   `-- replay_timing.py
`-- notebooks/
    |-- replay_timing_eda.py
    `-- deck_trends_eda.py
```

The existing `imitation_learning/notebooks/replay_timing.ipynb` is removed after its supported behavior is represented by the two Python commands.

## Configuration

### Replay timing extraction

`imitation_learning/cfg/extract_replay_timing.yaml`:

```yaml
extract:
  input: ../replay_episodes
  output: data/replay_timing
  date: latest
  force: false
```

- `input` is resolved relative to `imitation_learning` and contains dated replay archives.
- `output` is resolved relative to `imitation_learning`.
- `date` accepts `latest` or an explicit `M.D` value.
- `force: false` permits reuse of a valid existing extracted CSV; `true` rebuilds it.
- Progress is shown with the extraction command's standard progress display.
- Extraction errors are not capped by configuration. When errors exist, all error records are written to the dated error CSV and the terminal reports the total.

The command is:

```powershell
python -m extraction.replay_timing
```

### Replay timing EDA

`imitation_learning/cfg/replay_timing_eda.yaml`:

```yaml
analysis:
  input: data/replay_timing
  output: outputs/replay_timing
  date: latest
  score_filter:
    mode: min
    threshold: 1100
  clustering:
    n_clusters: 4
    random_state: 42
    n_init: 10
```

- `date` accepts `latest` or an explicit `M.D` value and selects an extracted player-timing CSV.
- `score_filter.mode` accepts `min`, `max`, or `avg`.
- A player row passes when the selected score is greater than or equal to `threshold`.
- Plot filenames and distribution/scatter styling are deliberately not configurable.
- Clustering remains configurable because it changes analytical results rather than presentation alone.

The command is:

```powershell
python notebooks/replay_timing_eda.py
```

## Extraction Responsibilities and Data Flow

`extraction/replay_timing.py` owns all access to raw replay archives:

1. locate the requested dated archive, resolving `latest` deterministically from available dates;
2. read its manifest and replay payloads;
3. calculate per-player timing and score fields using the notebook's existing definitions;
4. validate the extracted schema;
5. write or reuse the dated cache according to `force`;
6. write a dated error file only if extraction errors occurred.

The primary output is:

```text
data/replay_timing/<date>.player_timings.csv
```

It retains the current player-level fields, including date, episode ID, player index, team name, startup time, subsequent action total/count/mean, average/minimum/maximum/sum scores, the selected score value, and the score-filter flag where these are part of the existing extraction contract.

The maximum score continues to use the current two-player derivation:

```text
max_score = sum_score - min_score
```

When any replay cannot be extracted, all structured error records are written to:

```text
data/replay_timing/<date>.extraction_errors.csv
```

No error CSV is left for a successful extraction without errors. The extractor must not perform clustering or create plots.

## EDA Responsibilities and Data Flow

`notebooks/replay_timing_eda.py` reads only the extracted player-timing CSV. It must not scan replay JSON or ZIP archives.

The workflow is:

1. resolve the requested extracted date;
2. validate required columns and non-empty data;
3. aggregate player records into team timing metrics using the notebook's existing aggregation logic;
4. create the score-filtered team subset with the configured score mode and inclusive threshold;
5. fit one scaler and one KMeans model on all-team timing features;
6. assign all-team clusters from that fit and predict filtered-team clusters with the same scaler and cluster centers;
7. generate all six figures and all three tables in a temporary staging directory;
8. replace only the fixed, script-owned output files after every artifact has been generated successfully.

Using the same clustering model for both cohorts keeps cluster labels comparable. The filtered cohort must never fit a second independent clustering model.

## Fixed Outputs

### Figures

The replay timing EDA owns these fixed files under `outputs/replay_timing/`:

```text
all_teams_startup_distribution.png
all_teams_action_distribution.png
score_filtered_startup_distribution.png
score_filtered_action_distribution.png
all_teams_startup_vs_action.png
score_filtered_startup_vs_action.png
```

They preserve the notebook's two timing distributions for all teams, two timing distributions for score-filtered teams, and the startup-versus-action scatter plot for each cohort.

### Tables

The EDA owns these fixed files under `outputs/replay_timing/tables/`:

```text
all_team_timings.csv
score_filtered_team_timings.csv
cluster_summary.csv
```

- `all_team_timings.csv` contains the aggregated all-team metrics and assigned cluster.
- `score_filtered_team_timings.csv` contains the inclusive score-filtered subset and its comparable assigned cluster.
- `cluster_summary.csv` summarizes timing and membership for the globally fitted clusters.

The workflow does not create separate selected-date, extraction-summary, or cohort-summary tables.

## Output Replacement and Failure Handling

Both `replay_timing_eda.py` and `deck_trends_eda.py` use the same ownership rule:

- every chart/table is first written into a temporary staging location;
- the script validates that all expected artifacts were produced;
- only then are the fixed destination files replaced;
- an analysis or rendering failure leaves the previous complete output set intact;
- files in the output directory that are not in the script's fixed ownership list are not deleted or changed.

On a successful rerun, each owned file is overwritten with the result of the current configuration. Replacement is atomic per file; the staging-and-promote sequence prevents partial new artifacts from being published before the full result set is ready.

## Deck Trends Configuration Adjustment

Remove every `filename` key from `cfg/deck_trends_eda.yaml`. `notebooks/deck_trends_eda.py` owns the existing fixed filenames directly:

```text
archetype_share_and_win_rate.png
archetype_sankey.png
archetype_matchup_matrix.png
tables/line_chart_daily_metrics.csv
tables/matchup_matrix.csv
```

No deck trends calculation, filtering behavior, or visual content changes. Its output publication adopts the staging-and-replace behavior described above.

## Date Resolution

Date parsing is consistent between extraction and EDA:

- an explicit `M.D` selects exactly that dated input and produces a clear error if it is absent;
- `latest` compares parsed month/day values, not lexicographic filenames;
- extraction resolves among available replay archives;
- EDA resolves among available `.player_timings.csv` files;
- the resolved date is printed so runs remain auditable even though output filenames are fixed.

## Validation and Error Messages

Both commands fail early with actionable messages for:

- a missing input directory;
- no available archive or extracted CSV;
- an invalid date setting;
- an unsupported score mode;
- a missing or invalid required column;
- empty all-team data;
- an empty score-filtered cohort when a filtered artifact cannot be meaningfully generated;
- invalid clustering settings, including more clusters than available all-team samples.

EDA failure occurs before promotion, so these errors preserve the previous successful fixed outputs.

## Verification

Automated tests and command-level checks cover:

1. extraction date resolution for `latest` and explicit `M.D`;
2. EDA date resolution for `latest` and explicit `M.D`;
3. extracted cache schema and `force` reuse/rebuild behavior;
4. score modes `min`, `max`, and `avg`, including equality at the `>=` threshold;
5. KMeans is fitted once on all-team data and filtered clusters use the same scaler/model;
6. a successful EDA run creates exactly the six fixed PNGs and three fixed CSVs it owns;
7. a second successful run replaces those owned outputs;
8. a failed run preserves the previous complete outputs;
9. unrelated files in either EDA output directory remain unchanged;
10. the extraction error CSV exists only when errors occur and contains all captured errors;
11. deck trends no longer accepts or requires filename configuration;
12. the deleted replay timing notebook path is absent and project documentation names the new commands.

The implementation is also smoke-tested in the `orbit_wars` environment with representative available replay data, subject to local data availability.
