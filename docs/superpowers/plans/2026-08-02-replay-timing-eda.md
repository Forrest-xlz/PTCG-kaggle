# Replay Timing EDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible notebook that extracts team startup and subsequent action timing from the newest replay ZIP, exports reusable CSVs, and visualizes all teams plus a configurable score-filtered cohort with one global four-cluster K-Means model.

**Architecture:** Keep executable analysis in `imitation_learning/eda/replay_timing.ipynb` and generated tables in `imitation_learning/data/replay_timing/`. The notebook streams the newest ZIP, caches one row per replay player, aggregates one row per team and cohort, then fits a `log1p`/standardization/K-Means pipeline on all teams and reuses it for the filtered cohort. Move the existing deck notebook into the same EDA directory without changing the deck Python module.

**Tech Stack:** Python 3, Jupyter, `zipfile`, `json`, `csv`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nbformat`.

## Global Constraints

- Default score filter is `SCORE_MODE = "avg"` and `SCORE_THRESHOLD = 1100.0`.
- Supported score modes are exactly `avg`, `min`, and `max`.
- Fit `KMeans(n_clusters=4, random_state=42, n_init=10)` only on the all-team cohort.
- Use `log1p` plus `StandardScaler` internally; display original seconds on charts.
- Read the newest replay ZIP directly without extracting all members.
- Put notebooks under `imitation_learning/eda/` and timing CSVs under `imitation_learning/data/replay_timing/`.
- Keep chart text in English.
- Do not reorganize `imitation_learning/training/` in this change.

---

## File Map

- Move `imitation_learning/deck/deck_eda.ipynb` to `imitation_learning/eda/deck.ipynb`: centralize notebook artifacts.
- Create `imitation_learning/eda/replay_timing.ipynb`: parameters, extraction, validation, aggregation, clustering, export, and six charts.
- Modify `imitation_learning/requirements.txt`: declare notebook plotting and clustering dependencies.
- Generate at runtime `imitation_learning/data/replay_timing/<date>.player_timings.csv`: replay-player cache.
- Generate at runtime `imitation_learning/data/replay_timing/<date>.team_timings.csv`: final team/cohort table.

### Task 1: Centralize EDA notebooks and declare dependencies

**Files:**
- Move: `imitation_learning/deck/deck_eda.ipynb` -> `imitation_learning/eda/deck.ipynb`
- Modify: `imitation_learning/requirements.txt`

**Interfaces:**
- Consumes: existing deck notebook with project-root discovery.
- Produces: one stable EDA directory and an environment capable of executing the new notebook.

- [ ] **Step 1: Inspect the deck notebook for path assumptions**

Run:

```powershell
rg -n "PROJECT_ROOT|parents\[|deck_eda|imitation_learning/deck" imitation_learning/deck/deck_eda.ipynb
```

Expected: either root discovery independent of notebook location, or a small path reference that must be updated after the move.

- [ ] **Step 2: Move the notebook with Git-aware history preservation**

Run:

```powershell
New-Item -ItemType Directory -Force imitation_learning/eda
git mv imitation_learning/deck/deck_eda.ipynb imitation_learning/eda/deck.ipynb
```

If the notebook contains an explicit `imitation_learning/deck` self-path, update only that path to `imitation_learning/eda` using `nbformat`; retain cell order and outputs.

- [ ] **Step 3: Add required dependencies**

Append these declarations if absent:

```text
seaborn>=0.12
scikit-learn>=1.3
jupyter>=1.0
nbformat>=5.9
```

- [ ] **Step 4: Validate the moved notebook structure**

Run:

```powershell
python -c "import nbformat; n=nbformat.read('imitation_learning/eda/deck.ipynb', as_version=4); print(len(n.cells))"
```

Expected: prints a positive cell count without a schema exception.

- [ ] **Step 5: Commit the structural change**

```powershell
git add imitation_learning/eda/deck.ipynb imitation_learning/requirements.txt
git commit -m "refactor: centralize EDA notebooks"
```

### Task 2: Build deterministic archive, manifest, and timing extraction cells

**Files:**
- Create: `imitation_learning/eda/replay_timing.ipynb`

**Interfaces:**
- Consumes: repository root containing `replay_episodes/*.zip`.
- Produces: `find_latest_archive(replay_root: Path) -> tuple[str, Path]`, `load_manifest(archive_path: Path) -> pd.DataFrame`, `extract_player_timing(payload: dict, episode_id: str) -> list[dict]`, and `build_player_timings(archive_path: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]`.

- [ ] **Step 1: Scaffold the notebook as an analysis report**

Create cells in this order:

```text
# Replay Agent Timing EDA
## Context & Methods
### Parameters
### Locate Latest Replay Archive
### Load or Build Player Timing Cache
### Validate Extracted Timings
## Results
### Team Aggregation
### Timing Distributions
### Global Timing Clusters
## Takeaways and Caveats
```

The parameter cell must define:

```python
FORCE_REBUILD = False
SCORE_THRESHOLD = 1100.0
SCORE_MODE = "avg"  # avg, min, max
N_CLUSTERS = 4
RANDOM_STATE = 42
MAX_ERROR_EXAMPLES = 20
```

- [ ] **Step 2: Add a synthetic timing assertion before scanning real data**

Place this assertion immediately after the extraction helper cell so the timing contract remains executable:

```python
synthetic_payload = {
    "info": {"TeamNames": ["A", "B"]},
    "steps": [
        [
            {"observation": {"remainingOverageTime": 600.0}},
            {"observation": {"remainingOverageTime": 600.0}},
        ],
        [
            {"observation": {"remainingOverageTime": 598.0}},
            {"observation": {"remainingOverageTime": 599.0}},
        ],
        [
            {"observation": {"remainingOverageTime": 597.5}},
            {"observation": {"remainingOverageTime": 598.75}},
        ],
    ],
}
synthetic = extract_player_timing(synthetic_payload, "synthetic")
assert synthetic[0]["startup_time_seconds"] == 2.0
assert synthetic[0]["subsequent_time_seconds"] == 0.5
assert synthetic[0]["subsequent_action_count"] == 1
assert synthetic[1]["startup_time_seconds"] == 1.0
assert synthetic[1]["mean_step_time_seconds"] == 0.25
```

- [ ] **Step 3: Implement numeric date selection**

Use this contract:

```python
DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.zip$")

def find_latest_archive(replay_root: Path) -> tuple[str, Path]:
    candidates = []
    for path in replay_root.glob("*.zip"):
        match = DATE_RE.fullmatch(path.name)
        if match:
            month, day = map(int, match.groups())
            candidates.append(((month, day), f"{month}.{day}", path))
    if not candidates:
        raise FileNotFoundError(f"No M.D.zip replay archives found in {replay_root}")
    _, label, path = max(candidates, key=lambda item: item[0])
    return label, path
```

The setup cell locates `imitation_learning` by walking upward from `Path.cwd()` and then sets repository root to its parent, so execution works from either repository root or the notebook directory.

- [ ] **Step 4: Implement strict manifest loading and score modes**

`load_manifest` must find exactly one member whose basename is `manifest.csv`, require these columns, reject duplicate IDs, and derive `max_score`:

```python
MANIFEST_COLUMNS = {
    "episode_id", "avg_score", "min_score", "sum_score", "agent_count"
}

manifest["episode_id"] = manifest["episode_id"].astype(str)
manifest["max_score"] = manifest["sum_score"] - manifest["min_score"]
if SCORE_MODE not in {"avg", "min", "max"}:
    raise ValueError("SCORE_MODE must be 'avg', 'min', or 'max'")
manifest["score_value"] = manifest[f"{SCORE_MODE}_score"]
manifest["passes_score_filter"] = manifest["score_value"] > SCORE_THRESHOLD
```

- [ ] **Step 5: Implement replay-player timing extraction**

For each player index, collect finite remaining-time values in step order. Compute positive decreases with an epsilon of `1e-9`; the first positive decrease is startup and all later positive decreases are subsequent calls:

```python
deltas = [
    previous - current
    for previous, current in zip(values, values[1:])
    if previous - current > 1e-9
]
startup = deltas[0]
later = deltas[1:]
```

Return no row for missing/blank team names, fewer than two observations, or no positive decrease. Do not require a nonempty `action`, because an empty selection may still be a valid engine call.

- [ ] **Step 6: Stream JSON members and join the manifest**

`build_player_timings` opens one `ZipFile`, loops over `.json` members, uses the JSON `info.EpisodeId` or member stem, joins manifest values by string episode ID, and records at most `MAX_ERROR_EXAMPLES` dictionaries with `member` and `error`. It returns the complete DataFrame plus bounded diagnostics and does not retain replay payloads after each iteration.

- [ ] **Step 7: Validate notebook syntax without running the large scan**

Run:

```powershell
python -c "import ast,nbformat; n=nbformat.read('imitation_learning/eda/replay_timing.ipynb',as_version=4); [ast.parse(c.source) for c in n.cells if c.cell_type=='code']; print('syntax ok')"
```

Expected: `syntax ok`.

- [ ] **Step 8: Commit extraction logic**

```powershell
git add imitation_learning/eda/replay_timing.ipynb
git commit -m "feat: extract replay agent timing"
```

### Task 3: Add cache validation and team-level aggregation

**Files:**
- Modify: `imitation_learning/eda/replay_timing.ipynb`

**Interfaces:**
- Consumes: `build_player_timings`, selected date label, manifest-derived scores.
- Produces: `<date>.player_timings.csv`, `aggregate_teams(player_df: pd.DataFrame, mask: pd.Series, cohort: str) -> pd.DataFrame`, and combined `team_timings`.

- [ ] **Step 1: Define the cache schema and load/rebuild branch**

Use exactly these required columns:

```python
PLAYER_COLUMNS = [
    "date", "episode_id", "player_index", "team_name",
    "startup_time_seconds", "subsequent_time_seconds",
    "subsequent_action_count", "mean_step_time_seconds",
    "avg_score", "min_score", "max_score", "sum_score",
    "score_value", "passes_score_filter",
]
```

If the player CSV exists and `FORCE_REBUILD` is false, load it and require every column. Otherwise build, order columns, create the output directory, and save with `index=False`.

- [ ] **Step 2: Add data-quality assertions**

The validation cell must assert:

```python
assert not player_timings.empty
assert player_timings["team_name"].str.strip().ne("").all()
assert player_timings["startup_time_seconds"].ge(0).all()
assert player_timings["subsequent_time_seconds"].ge(0).all()
assert player_timings["subsequent_action_count"].ge(0).all()
assert player_timings["player_index"].isin([0, 1]).all()
```

Display row count, episode count, team count, skipped-member count, and the selected archive/date.

- [ ] **Step 3: Implement action-weighted team aggregation**

Use this aggregation contract:

```python
def aggregate_teams(player_df: pd.DataFrame, mask: pd.Series, cohort: str) -> pd.DataFrame:
    selected = player_df.loc[mask].copy()
    grouped = selected.groupby("team_name", as_index=False).agg(
        replay_count=("episode_id", "nunique"),
        startup_time_mean_seconds=("startup_time_seconds", "mean"),
        subsequent_time_seconds=("subsequent_time_seconds", "sum"),
        subsequent_action_count=("subsequent_action_count", "sum"),
    )
    grouped = grouped[grouped["subsequent_action_count"] > 0].copy()
    grouped["mean_step_time_seconds"] = (
        grouped["subsequent_time_seconds"]
        / grouped["subsequent_action_count"]
    )
    grouped.insert(0, "cohort", cohort)
    return grouped
```

Build `all_teams` with an all-true mask and `filtered_teams` from `passes_score_filter`. Concatenate them only after clustering assigns labels.

- [ ] **Step 4: Add an aggregation reasonableness check**

```python
expected_time = player_timings["subsequent_time_seconds"].sum()
observed_time = all_teams["subsequent_time_seconds"].sum()
assert np.isclose(expected_time, observed_time)
assert all_teams["team_name"].is_unique
assert filtered_teams["team_name"].is_unique
```

- [ ] **Step 5: Commit cache and aggregation cells**

```powershell
git add imitation_learning/eda/replay_timing.ipynb
git commit -m "feat: aggregate replay timings by team"
```

### Task 4: Fit one global clustering model and export the final team CSV

**Files:**
- Modify: `imitation_learning/eda/replay_timing.ipynb`

**Interfaces:**
- Consumes: `all_teams`, `filtered_teams`, `N_CLUSTERS`, `RANDOM_STATE`.
- Produces: fitted `StandardScaler`, fitted `KMeans`, `cluster_centers_seconds`, and `<date>.team_timings.csv`.

- [ ] **Step 1: Add a transform helper and pre-fit assertions**

```python
FEATURE_COLUMNS = ["startup_time_mean_seconds", "mean_step_time_seconds"]

def log_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Timing features must be finite and nonnegative")
    return np.log1p(values)

if len(all_teams) < N_CLUSTERS:
    raise ValueError(
        f"Need at least {N_CLUSTERS} all-cohort teams; found {len(all_teams)}"
    )
```

- [ ] **Step 2: Fit globally and predict both cohorts**

```python
scaler = StandardScaler()
all_scaled = scaler.fit_transform(log_features(all_teams))
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10,
)
all_teams["cluster"] = kmeans.fit_predict(all_scaled)
filtered_teams["cluster"] = kmeans.predict(
    scaler.transform(log_features(filtered_teams))
) if len(filtered_teams) else pd.Series(dtype="int64")
cluster_centers_seconds = np.expm1(
    scaler.inverse_transform(kmeans.cluster_centers_)
)
```

- [ ] **Step 3: Export long-form team timings**

Concatenate cohorts, order by `cohort`, `cluster`, and `team_name`, and write:

```python
team_timings = pd.concat([all_teams, filtered_teams], ignore_index=True)
team_timings["score_mode"] = SCORE_MODE
team_timings["score_threshold"] = SCORE_THRESHOLD
team_timings.to_csv(team_cache_path, index=False)
```

The exported columns must include cohort, team name, replay/action counts, startup/step timing, cluster, score mode, and threshold.

- [ ] **Step 4: Verify model reuse explicitly**

```python
if len(filtered_teams):
    repeated = kmeans.predict(scaler.transform(log_features(filtered_teams)))
    assert np.array_equal(repeated, filtered_teams["cluster"].to_numpy())
assert len(cluster_centers_seconds) == N_CLUSTERS
```

- [ ] **Step 5: Commit clustering and export**

```powershell
git add imitation_learning/eda/replay_timing.ipynb
git commit -m "feat: cluster team timing profiles"
```

### Task 5: Add six English figures and concise result tables

**Files:**
- Modify: `imitation_learning/eda/replay_timing.ipynb`

**Interfaces:**
- Consumes: both team cohorts, global cluster labels, and centers in seconds.
- Produces: four histograms, two scatter plots, a cohort summary table, and a cluster-center table.

- [ ] **Step 1: Define a restrained fixed palette and chart helpers**

```python
CLUSTER_COLORS = {
    0: "#2F6BFF",
    1: "#D97706",
    2: "#708238",
    3: "#C24170",
}
sns.set_theme(style="whitegrid", context="notebook")
```

`plot_histogram(frame, column, title, xlabel)` must count one row per team, use positive values, show `Teams` on the y-axis, use logarithmic x scale when the minimum is positive, and put the team count in a subtitle or annotation.

- [ ] **Step 2: Render all-team distributions**

Render separate figures for:

```text
All Teams: Startup Time Distribution
All Teams: Mean Subsequent Action Time Distribution
```

Use `startup_time_mean_seconds` and `mean_step_time_seconds`; label both x-axes in seconds.

- [ ] **Step 3: Render filtered distributions with dynamic titles**

Construct:

```python
FILTER_LABEL = f"{SCORE_MODE}_score > {SCORE_THRESHOLD:g}"
```

Render separate startup and mean-step histograms titled with `FILTER_LABEL`. If the filtered cohort is empty, display a concise message and do not call the histogram function.

- [ ] **Step 4: Render the all-team clustered scatter**

Plot one team per point with:

```text
x = mean_step_time_seconds
y = startup_time_mean_seconds
```

Use cluster colors, original seconds, logarithmic axes when positive, alpha below 1, and black `X` markers for inverse-transformed global centers. Title: `All Teams: Startup vs Mean Subsequent Action Time`.

- [ ] **Step 5: Render the filtered scatter with the same centers**

Reuse `CLUSTER_COLORS`, assigned labels, and `cluster_centers_seconds`; do not call `.fit` or `.fit_predict`. Title includes `FILTER_LABEL`. If empty, display a concise message.

- [ ] **Step 6: Display bounded summary tables and caveats**

Show:

- cohort, team count, median startup, median step time, total replays, total actions;
- cluster, center startup seconds, center step seconds, all-team count, filtered-team count.

The final markdown cell states that timing clusters suggest behavioral profiles but do not prove whether an agent is rule-based, neural, search-based, or hybrid. It also explains that `avg` and `max` are replay-level filters rather than exact player-score mappings.

- [ ] **Step 7: Commit visualization cells**

```powershell
git add imitation_learning/eda/replay_timing.ipynb
git commit -m "feat: visualize replay timing clusters"
```

### Task 6: Execute and verify the complete notebook

**Files:**
- Verify: `imitation_learning/eda/replay_timing.ipynb`
- Verify generated: `imitation_learning/data/replay_timing/<date>.player_timings.csv`
- Verify generated: `imitation_learning/data/replay_timing/<date>.team_timings.csv`

**Interfaces:**
- Consumes: newest replay archive and all notebook cells.
- Produces: executed notebook outputs and validated CSV schemas.

- [ ] **Step 1: Run the notebook top-to-bottom**

Run from `imitation_learning` so root discovery is exercised:

```powershell
python -m jupyter nbconvert --execute --to notebook --inplace eda/replay_timing.ipynb --ExecutePreprocessor.timeout=-1
```

Expected: exit code 0. The first run may take several minutes because it streams the newest ZIP; later runs load the player CSV cache.

- [ ] **Step 2: Validate generated CSV schemas and row grain**

Run:

```powershell
python -c "import pandas as pd; p=pd.read_csv('data/replay_timing/7.27.player_timings.csv'); t=pd.read_csv('data/replay_timing/7.27.team_timings.csv'); assert not p.empty and not t.empty; assert not t.duplicated(['cohort','team_name']).any(); assert set(t.cohort)=={'all','score_filtered'}; print(p.shape,t.shape)"
```

Replace `7.27` with the selected date printed by the notebook if a newer archive exists.

- [ ] **Step 3: Inspect notebook outputs**

Confirm all six expected chart titles are present, filtered titles show the configured score expression, cluster centers are identical across the two scatter plots, and chart axes are labeled in seconds.

- [ ] **Step 4: Check repository integrity**

Run:

```powershell
git diff --check
git status --short
```

Expected: generated `data/` CSVs remain ignored; only intended notebook/dependency changes appear.

- [ ] **Step 5: Commit executed notebook outputs if intentionally retained**

```powershell
git add imitation_learning/eda/replay_timing.ipynb
git commit -m "docs: record replay timing EDA outputs"
```

If the repository convention is to keep notebooks without heavy outputs, clear only large cell outputs with `nbformat`, revalidate the notebook schema, and commit the clean notebook instead.
