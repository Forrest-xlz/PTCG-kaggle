# Replay Timing EDA Design

## Goal

Create a reproducible notebook that analyzes agent startup time and subsequent
per-action time from the newest replay archive. The analysis operates at team
grain, compares all teams with a strict 1100+ cohort, and uses one global
four-cluster K-Means model for both scatter plots.

## Project Layout

```text
imitation_learning/
├─ deck/
│  ├─ extract.py
│  └─ analysis.py
├─ eda/
│  ├─ deck.ipynb
│  └─ replay_timing.ipynb
└─ data/
   └─ replay_timing/
      ├─ <date>.player_timings.csv
      └─ <date>.team_timings.csv
```

Move `imitation_learning/deck/deck_eda.ipynb` to
`imitation_learning/eda/deck.ipynb`. Do not reorganize the training package in
this change.

## Inputs and Date Selection

- Search the repository-level `replay_episodes/` directory for ZIP archives.
- Parse filenames as numeric month/day pairs and select the latest date
  numerically. Lexicographic filename ordering must not be used.
- Read JSON replay members and the archive's single `manifest.csv` directly
  from the ZIP without extracting the archive to disk.
- Match JSON and manifest rows by episode ID.
- Read team identity from `payload["info"]["TeamNames"][player_index]`.

The notebook parameters cell exposes:

- `FORCE_REBUILD = False`
- `SCORE_THRESHOLD = 1100.0`
- `N_CLUSTERS = 4`
- `RANDOM_STATE = 42`

## Timing Definition

For each replay and each player, inspect the sequence of
`observation.remainingOverageTime` values. The time spent by an agent call is
the positive decrease from that player's previous observed value.

- **Startup time:** the first positive decrease, corresponding to the initial
  deck submission/model startup call.
- **Subsequent action time:** every later positive decrease for the same
  player.
- Ignore zero and negative differences, which do not represent consumed agent
  time.
- Keep an episode-player row only when a team name and startup time can be
  resolved. Record missing or malformed members in a bounded diagnostic
  summary instead of aborting the entire archive scan.

## Cached Player Table

Write `imitation_learning/data/replay_timing/<date>.player_timings.csv` with one
row per replay player. It is both an auditable intermediate table and the
cache that prevents rescanning a large ZIP on every notebook run.

Required columns:

- `date`
- `episode_id`
- `player_index`
- `team_name`
- `startup_time_seconds`
- `subsequent_time_seconds`
- `subsequent_action_count`
- `mean_step_time_seconds`
- `avg_score`
- `min_score`
- `sum_score`
- `is_score_1100_plus`

`is_score_1100_plus` is computed as `min_score > SCORE_THRESHOLD`. This strict
definition guarantees that both teams in the replay exceed the threshold,
because the manifest does not map its two individual scores back to player
indices.

When the cache exists and `FORCE_REBUILD` is false, load it after checking the
required columns. Rebuild when forced or when the cache schema is incomplete.

## Team Aggregation

Create two cohorts independently:

1. `all`: every valid episode-player row.
2. `score_1100_plus`: only rows whose replay has `min_score > 1100`.

Within each cohort, group by exact `team_name`:

- `replay_count`: number of replay appearances.
- `startup_time_mean_seconds`: arithmetic mean of per-replay startup times.
- `subsequent_action_count`: total subsequent calls across replays.
- `mean_step_time_seconds`: total subsequent time divided by total subsequent
  calls. This is action-weighted rather than an unweighted mean of replay
  means.

Each team has equal weight in charts and K-Means regardless of its replay
count. Retain replay and action counts in the exported table so readers can
judge noisy low-volume estimates.

Write `imitation_learning/data/replay_timing/<date>.team_timings.csv` as the
final reusable analysis table. Store both cohorts in long form with a `cohort`
column and include the assigned global cluster.

## Clustering

Fit the clustering pipeline only on complete team rows from the `all` cohort:

1. Apply `log1p` to startup and mean step time.
2. Standardize the two transformed features with `StandardScaler`.
3. Fit `KMeans(n_clusters=4, random_state=42, n_init=10)`.

Use the same fitted scaler and K-Means model to predict clusters for the
1100+ cohort. Do not refit on the subset. Plot coordinates remain in original
seconds; transformation is internal to clustering.

Cluster numbers are descriptive identifiers only and must not be labeled as
definitive rule-based, neural, search, or hybrid implementations.

## Notebook Flow and Figures

The notebook is an analysis report with concise English labels to avoid font
issues:

1. Goal, assumptions, imports, and parameters.
2. Locate the latest archive and explain selected date.
3. Load or build the player timing cache.
4. Validate and summarize player timing data.
5. Aggregate both team cohorts and fit the global clustering pipeline.
6. Export the final team table.
7. Render six figures:
   - all-team startup-time histogram;
   - all-team mean-step-time histogram;
   - 1100+ startup-time histogram;
   - 1100+ mean-step-time histogram;
   - all-team startup-versus-step scatter, colored by global cluster;
   - 1100+ startup-versus-step scatter using the same global clusters.
8. Display compact cluster-center and cohort-summary tables.
9. State interpretation caveats.

Histogram y-axes count unique teams. Scatter points are teams, not individual
replays. Cluster centers are inverse-transformed and shown in seconds on the
all-team scatter; the 1100+ scatter uses the same centers as references.

## Visualization Rules

- Use a white background, quiet grid lines, dark text, and an explicit
  four-color palette.
- Use neutral descriptive titles and include cohort/team counts in subtitles
  or nearby text.
- Because both timing measures are long-tailed, use log-scaled x-axes on timing
  histograms and log-scaled x/y axes on scatter plots when all displayed
  values are positive. Labels continue to report seconds.
- Include cluster markers in addition to color for cluster centers.
- Do not infer implementation type from cluster membership without external
  evidence.

## Error Handling and Validation

- Fail clearly when no dated ZIP archive or no unique `manifest.csv` exists.
- Validate required manifest columns before scanning replay members.
- Reject duplicate manifest episode IDs.
- Bound malformed-member diagnostics to avoid enormous notebook output.
- Verify nonnegative timing values, nonempty team names, and positive action
  counts for rows used in mean-step plots and clustering.
- Require at least four complete all-cohort teams before K-Means fitting.
- Execute the notebook top-to-bottom when the local Python environment has the
  required dependencies and enough time to scan the newest archive. If local
  execution is unavailable, validate notebook JSON and provide the exact
  execution command.

## Dependencies

Use packages already appropriate for the project and notebook workflow:

- Python standard library: `csv`, `json`, `re`, `zipfile`, `pathlib`.
- `numpy`, `pandas`, `matplotlib`, `seaborn`.
- `scikit-learn` for `StandardScaler` and `KMeans`.

No command-line interface or additional YAML configuration is added. The
notebook remains self-contained apart from its replay input and generated CSV
files.
