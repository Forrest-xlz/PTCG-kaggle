# Replay Timing Score-Filtered Team Table Design

## Goal

Add one reader-facing cell to `imitation_learning/eda/replay_timing.ipynb`
that identifies every score-filtered team and shows its score and timing
profile in a compact table.

## Placement and Inputs

Insert the cell after global cluster assignment and cohort summary, before the
distribution charts. It will reuse these existing notebook objects:

- `player_timings` and `filtered_mask` for the active replay-level score;
- `filtered_teams` for team-level timing aggregation and global cluster labels;
- `SCORE_MODE` and `SCORE_THRESHOLD` for visible cohort context.

No replay archive is rescanned and no existing extraction, aggregation,
clustering, plot, or CSV logic changes.

## Table Grain and Columns

The table has one row per exact `team_name` in the score-filtered cohort and
contains:

- `team_name`;
- `replay_count`;
- `score_mode`;
- `mean_score`;
- `mean_startup_time_seconds`;
- `mean_action_time_seconds`;
- `cluster`.

`mean_score` is the arithmetic mean of the active replay-level `score_value`
across that team's filtered replay appearances. It is not an individual-player
score because `manifest.csv` does not map its two participant scores back to
player indices. `mean_startup_time_seconds` and `mean_action_time_seconds`
reuse the existing team-level definitions, so the table matches the plotted
team points exactly.

Rows are sorted by `mean_score` descending and then `team_name` ascending for
deterministic tie handling. Numeric values are rounded only for display; the
underlying frames retain full precision.

## Empty Cohort and Validation

If no teams satisfy the active score filter, print the same concise empty
cohort message used elsewhere instead of raising an error. Otherwise assert
that team names are unique and that the table contains exactly the same teams
as `filtered_teams`.

Validate the notebook as JSON and, when the local kernel has the required
archive and dependencies, execute it top-to-bottom. If execution is not
possible, report the exact missing dependency or data input.
