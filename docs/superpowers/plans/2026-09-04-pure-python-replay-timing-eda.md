# Pure Python Replay Timing EDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the replay timing notebook with separate extraction and pure Python EDA commands, while giving Replay Timing and Deck Trends reliable fixed-output publication.

**Architecture:** `extraction/replay_timing.py` is the only component that reads replay archives and emits dated player-level caches. `notebooks/replay_timing_eda.py` reads one cache, aggregates teams, applies an inclusive score filter, fits one global clustering model, stages six plots and three tables, and then promotes the complete artifact set. Deck Trends keeps its existing calculations but adopts fixed filenames and the same staged publication boundary.

**Tech Stack:** Python 3, pathlib, dataclasses, zipfile, json, pandas, NumPy, PyYAML, matplotlib, seaborn, scikit-learn, pytest

**Spec:** `docs/superpowers/specs/2026-09-04-pure-python-replay-timing-eda-design.md`

## Global Constraints

- Run Python and pytest with `C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe`.
- Extraction YAML contains only `input`, `output`, `date`, and `force`; do not add `max_error_examples` or `progress_every`.
- Replay Timing and Deck Trends output filenames are constants in Python, never YAML values.
- Replay score filtering is always inclusive: selected score `>= threshold`.
- Fit the scaler and KMeans model once on all-team data; predict the filtered subset with that same fit.
- Generate every owned EDA artifact before replacing any existing owned destination.
- Never delete or replace unrelated output files.
- Preserve existing timing definitions, score formulas, plot content, and Deck Trends analytical behavior.

---

### Task 1: Replay Timing Extraction Command

**Files:**
- Create: `imitation_learning/extraction/replay_timing.py`
- Create: `imitation_learning/cfg/extract_replay_timing.yaml`
- Create: `imitation_learning/tests/test_replay_timing_extract.py`

**Interfaces:**
- Produces: `ExtractionSettings(input: Path, output: Path, date: str, force: bool)`
- Produces: `load_settings(path: Path, project_root: Path = PROJECT_ROOT) -> ExtractionSettings`
- Produces: `parse_month_day(value: str) -> tuple[int, int]`
- Produces: `resolve_archive(input_dir: Path, date: str) -> tuple[str, Path]`
- Produces: `extract_player_timings(payload: dict[str, Any], date: str) -> list[dict[str, Any]]`
- Produces: `extract_archive(archive_path: Path, output_dir: Path, date: str, force: bool) -> dict[str, Any]`
- Produces: dated `<date>.player_timings.csv` and conditional `<date>.extraction_errors.csv`

- [ ] **Step 1: Write failing settings and date-resolution tests**

Add tests that build YAML with exactly the four supported fields and create fake archives whose names prove parsed ordering rather than lexical ordering:

```python
def test_settings_have_only_the_supported_controls(tmp_path: Path) -> None:
    config = tmp_path / "extract.yaml"
    config.write_text(
        "extract:\n"
        "  input: archives\n"
        "  output: timing\n"
        "  date: latest\n"
        "  force: false\n",
        encoding="utf-8",
    )
    settings = load_settings(config, project_root=tmp_path)
    assert settings == ExtractionSettings(
        input=(tmp_path / "archives").resolve(),
        output=(tmp_path / "timing").resolve(),
        date="latest",
        force=False,
    )


def test_resolve_archive_uses_parsed_latest_and_explicit_date(tmp_path: Path) -> None:
    for name in ("8.9.zip", "8.15.zip", "7.31.zip"):
        (tmp_path / name).touch()
    assert resolve_archive(tmp_path, "latest") == ("8.15", tmp_path / "8.15.zip")
    assert resolve_archive(tmp_path, "8.9") == ("8.9", tmp_path / "8.9.zip")
    with pytest.raises(FileNotFoundError, match="8.10"):
        resolve_archive(tmp_path, "8.10")
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_replay_timing_extract.py -q
```

Expected: collection/import failure because `extraction.replay_timing` does not exist.

- [ ] **Step 3: Implement config loading and deterministic date resolution**

Create the dataclass and validation. Reject unknown keys so the removed logging options cannot silently return:

```python
@dataclass(frozen=True)
class ExtractionSettings:
    input: Path
    output: Path
    date: str
    force: bool


def resolve_archive(input_dir: Path, date: str) -> tuple[str, Path]:
    archives = {path.stem: path for path in input_dir.glob("*.zip")}
    if not archives:
        raise FileNotFoundError(f"No replay archives found in {input_dir}")
    resolved = max(archives, key=parse_month_day) if date == "latest" else date
    if resolved not in archives:
        raise FileNotFoundError(f"Replay archive for {resolved} not found in {input_dir}")
    return resolved, archives[resolved]
```

Add the approved config with no progress/error-example controls.

- [ ] **Step 4: Write failing extraction, cache, and error-file tests**

Create a minimal ZIP with `manifest.csv` and replay JSON fixtures shaped like the real archive. Assert:

```python
def test_extract_archive_writes_player_rows_and_reuses_valid_cache(tmp_path: Path) -> None:
    archive = write_timing_archive(tmp_path / "8.15.zip")
    summary = extract_archive(archive, tmp_path / "out", "8.15", force=False)
    rows = pd.read_csv(tmp_path / "out" / "8.15.player_timings.csv")
    assert summary["status"] == "written"
    assert len(rows) == 2
    assert rows.loc[0, "max_score"] == rows.loc[0, "sum_score"] - rows.loc[0, "min_score"]
    assert extract_archive(archive, tmp_path / "out", "8.15", force=False)["status"] == "skipped"


def test_errors_are_all_written_and_clean_run_removes_stale_error_file(tmp_path: Path) -> None:
    archive = write_timing_archive(tmp_path / "8.15.zip", malformed_members=2)
    extract_archive(archive, tmp_path / "out", "8.15", force=True)
    errors = pd.read_csv(tmp_path / "out" / "8.15.extraction_errors.csv")
    assert len(errors) == 2
    clean_archive = write_timing_archive(tmp_path / "8.15.zip", malformed_members=0)
    extract_archive(clean_archive, tmp_path / "out", "8.15", force=True)
    assert not (tmp_path / "out" / "8.15.extraction_errors.csv").exists()
```

- [ ] **Step 5: Port the notebook extraction logic without analytical code**

Move the existing manifest/replay parsing and timing calculations into focused helpers. Keep the current output columns and formulas. `main()` must load the fixed config path, resolve the date, call `extract_archive`, and print the resolved date and summary. Do not import plotting or clustering libraries.

- [ ] **Step 6: Run extraction tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_replay_timing_extract.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit the extraction command**

```powershell
git add imitation_learning/extraction/replay_timing.py imitation_learning/cfg/extract_replay_timing.yaml imitation_learning/tests/test_replay_timing_extract.py
git commit -m "feat: extract replay timing data"
```

---

### Task 2: Replay Timing Analysis and Inclusive Score Filtering

**Files:**
- Create: `imitation_learning/notebooks/replay_timing_eda.py`
- Create: `imitation_learning/cfg/replay_timing_eda.yaml`
- Replace: `imitation_learning/tests/test_replay_timing_notebook.py`

**Interfaces:**
- Consumes: `<date>.player_timings.csv` from Task 1
- Produces: `AnalysisSettings`, `ScoreFilter`, and `ClusteringSettings` dataclasses
- Produces: `load_settings(path: Path) -> AnalysisSettings`
- Produces: `resolve_timing_csv(input_dir: Path, date: str) -> tuple[str, Path]`
- Produces: `aggregate_team_timings(players: pd.DataFrame) -> pd.DataFrame`
- Produces: `filter_team_timings(teams: pd.DataFrame, score_filter: ScoreFilter) -> pd.DataFrame`
- Produces: `assign_timing_clusters(teams: pd.DataFrame, filtered: pd.DataFrame, settings: ClusteringSettings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Replace the notebook-content test with config and path tests**

The replacement test imports the new Python module and verifies that style/filename options are absent:

```python
def test_replay_timing_eda_config_is_analysis_only() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["analysis"]
    assert set(raw) == {"input", "output", "date", "score_filter", "clustering"}
    assert set(raw["score_filter"]) == {"mode", "threshold"}
    assert set(raw["clustering"]) == {"n_clusters", "random_state", "n_init"}


def test_pure_python_replay_timing_files_exist() -> None:
    assert (PROJECT_ROOT / "notebooks" / "replay_timing_eda.py").is_file()
    assert (PROJECT_ROOT / "cfg" / "extract_replay_timing.yaml").is_file()
    assert (PROJECT_ROOT / "cfg" / "replay_timing_eda.yaml").is_file()
```

- [ ] **Step 2: Write failing date, inclusive-filter, and aggregation tests**

```python
def test_resolve_timing_csv_handles_latest_and_explicit_dates(tmp_path: Path) -> None:
    for name in ("8.9.player_timings.csv", "8.15.player_timings.csv"):
        (tmp_path / name).touch()
    assert resolve_timing_csv(tmp_path, "latest")[0] == "8.15"
    assert resolve_timing_csv(tmp_path, "8.9")[0] == "8.9"


@pytest.mark.parametrize(
    ("mode", "score_column"),
    [("min", "min_score"), ("max", "max_score"), ("avg", "avg_score")],
)
def test_score_filter_is_inclusive(mode: str, score_column: str) -> None:
    teams = team_frame_with_scores(equal_to=1100, below=1099)
    filtered = filter_team_timings(teams, ScoreFilter(mode=mode, threshold=1100))
    assert filtered["team_name"].tolist() == ["Equal"]
    assert filtered.loc[filtered.index[0], "score_value"] == teams.loc[0, score_column]


def test_team_aggregation_preserves_timing_and_score_metrics() -> None:
    teams = aggregate_team_timings(player_timing_frame())
    assert list(teams["team_name"]) == ["Alpha", "Beta"]
    assert {"mean_startup_time_seconds", "mean_action_time_seconds", "avg_score", "min_score", "max_score"} <= set(teams)
```

- [ ] **Step 3: Run tests and confirm missing implementation failures**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_replay_timing_notebook.py -q
```

Expected: failures for the missing module/functions and still-present notebook.

- [ ] **Step 4: Implement settings, date resolution, aggregation, and filtering**

Use explicit mode-to-column mapping and reject unsupported modes:

```python
SCORE_COLUMNS = {"min": "min_score", "max": "max_score", "avg": "avg_score"}


def filter_team_timings(teams: pd.DataFrame, score_filter: ScoreFilter) -> pd.DataFrame:
    try:
        score_column = SCORE_COLUMNS[score_filter.mode]
    except KeyError as error:
        raise ValueError(f"Unsupported score mode: {score_filter.mode}") from error
    result = teams.copy()
    result["score_value"] = result[score_column]
    return result.loc[result["score_value"].ge(score_filter.threshold)].copy()
```

Port team aggregation from the notebook, including the score summary merged one-to-one on `team_name`. Validate required columns and empty inputs before aggregating.

- [ ] **Step 5: Write failing single-fit clustering tests**

Monkeypatch module-level `StandardScaler` and `KMeans` spies, then assert exactly one fit and reuse via predict:

```python
def test_filtered_clusters_reuse_the_global_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    scaler_spy, kmeans_spy = install_cluster_spies(monkeypatch)
    all_result, filtered_result, summary = assign_timing_clusters(
        all_team_frame(), filtered_team_frame(), ClusteringSettings(2, 42, 10)
    )
    assert scaler_spy.fit_calls == 1
    assert kmeans_spy.fit_predict_calls == 1
    assert kmeans_spy.predict_calls == 1
    assert set(filtered_result["cluster"]) <= set(all_result["cluster"])
    assert set(summary["cluster"]) == set(all_result["cluster"])
```

- [ ] **Step 6: Implement global clustering and summary**

Fit `StandardScaler` on the two established timing features, fit KMeans once, predict filtered rows with the same objects, and summarize cluster counts and timing means. Reject `n_clusters < 1`, `n_clusters > len(teams)`, non-positive `n_init`, and an empty filtered cohort with clear errors.

- [ ] **Step 7: Run analysis unit tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_replay_timing_notebook.py -q
```

Expected: all analysis and configuration tests pass.

- [ ] **Step 8: Commit analysis logic and configuration**

```powershell
git add imitation_learning/notebooks/replay_timing_eda.py imitation_learning/cfg/replay_timing_eda.yaml imitation_learning/tests/test_replay_timing_notebook.py
git commit -m "feat: analyze replay timing data"
```

---

### Task 3: Complete and Transactional Replay Timing Artifacts

**Files:**
- Modify: `imitation_learning/notebooks/replay_timing_eda.py`
- Modify: `imitation_learning/tests/test_replay_timing_notebook.py`

**Interfaces:**
- Consumes: analysis frames/functions from Task 2
- Produces: `FIGURE_FILENAMES: tuple[str, ...]` containing six fixed names
- Produces: `TABLE_FILENAMES: tuple[str, ...]` containing three fixed names
- Produces: `publish_artifacts(staging: Path, output: Path) -> None`
- Produces: `run(settings: AnalysisSettings) -> tuple[str, Path]`

- [ ] **Step 1: Write failing complete-output tests**

Call `run()` against a synthetic extracted CSV and small valid cluster count:

```python
def test_run_writes_six_fixed_figures_and_three_tables(tmp_path: Path) -> None:
    settings = write_analysis_fixture(tmp_path)
    resolved_date, output = run(settings)
    assert resolved_date == "8.15"
    assert {path.name for path in output.glob("*.png")} == set(FIGURE_FILENAMES)
    assert {path.name for path in (output / "tables").glob("*.csv")} == set(TABLE_FILENAMES)
```

Also load each CSV and assert the expected table-specific columns, especially `cluster` in both team tables and the cluster summary.

- [ ] **Step 2: Run the artifact test and confirm it fails**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_replay_timing_notebook.py::test_run_writes_six_fixed_figures_and_three_tables -q
```

Expected: failure because plots/publication are not implemented.

- [ ] **Step 3: Implement the six plots and three fixed tables**

Port the notebook's distribution and scatter rendering into small functions. Use module constants:

```python
FIGURE_FILENAMES = (
    "all_teams_startup_distribution.png",
    "all_teams_action_distribution.png",
    "score_filtered_startup_distribution.png",
    "score_filtered_action_distribution.png",
    "all_teams_startup_vs_action.png",
    "score_filtered_startup_vs_action.png",
)
TABLE_FILENAMES = (
    "all_team_timings.csv",
    "score_filtered_team_timings.csv",
    "cluster_summary.csv",
)
```

Keep plot styling as Python constants/defaults. Close every matplotlib figure after saving. Write CSVs with `index=False` and `encoding="utf-8-sig"`.

- [ ] **Step 4: Write failing overwrite, failure-preservation, and unrelated-file tests**

```python
def test_success_replaces_owned_outputs_and_preserves_unrelated_file(tmp_path: Path) -> None:
    output, staging = prepared_output_and_staging(tmp_path)
    unrelated = output / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    (output / FIGURE_FILENAMES[0]).write_bytes(b"old")
    publish_artifacts(staging, output)
    assert (output / FIGURE_FILENAMES[0]).read_bytes() == b"new"
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_generation_failure_preserves_previous_complete_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings, snapshots = prepared_previous_outputs(tmp_path)
    monkeypatch.setattr(replay_timing_eda, "save_scatter", raise_render_error)
    with pytest.raises(RuntimeError, match="render"):
        run(settings)
    assert snapshot_owned_outputs(settings.output) == snapshots
```

- [ ] **Step 5: Implement staging and promotion**

Use `tempfile.TemporaryDirectory(dir=output.parent)` for generation. Validate all nine staged paths before promotion. Ensure destination parent directories exist, then use `os.replace(staged_file, destination)` for each fixed owned file. Do not enumerate/delete the destination directory and do not touch files outside the constants.

- [ ] **Step 6: Run all replay timing EDA tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_replay_timing_notebook.py -q
```

Expected: all replay timing EDA tests pass.

- [ ] **Step 7: Commit fixed, transactional artifact generation**

```powershell
git add imitation_learning/notebooks/replay_timing_eda.py imitation_learning/tests/test_replay_timing_notebook.py
git commit -m "feat: generate replay timing EDA artifacts"
```

---

### Task 4: Deck Trends Fixed Filenames and Staged Publication

**Files:**
- Modify: `imitation_learning/cfg/deck_trends_eda.yaml`
- Modify: `imitation_learning/notebooks/deck_trends_eda.py`
- Modify: `imitation_learning/tests/test_deck_trends_eda.py`

**Interfaces:**
- Produces: `FIGURE_FILENAMES` mapping for `line_chart`, `sankey`, and `matchup_matrix`
- Produces: fixed table names already established by the script
- Produces: complete staged generation before owned-file replacement

- [ ] **Step 1: Change the config test to require no filename fields**

```python
def test_eda_config_defines_three_independent_charts_without_filenames() -> None:
    analysis = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["analysis"]
    line = _chart_settings(analysis, "line_chart", mirrors=True)
    sankey = _chart_settings(analysis, "sankey", mirrors=False)
    matrix = _chart_settings(analysis, "matchup_matrix", mirrors=True)
    assert line["score_filter"].mode == "all"
    assert sankey["score_filter"].mode == "min"
    assert matrix["score_filter"].mode == "min"
    assert all("filename" not in settings for settings in (line, sankey, matrix))
```

- [ ] **Step 2: Add a failing atomic-generation regression test**

Patch `_save_sankey` to raise after the line chart has rendered into staging, seed all five previous owned outputs plus an unrelated file, call `main()`, and assert every previous owned byte snapshot and the unrelated file remain unchanged.

- [ ] **Step 3: Run focused Deck Trends tests and confirm failures**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_deck_trends_eda.py -q
```

Expected: filename-removal and staging tests fail against current behavior.

- [ ] **Step 4: Remove YAML filenames and use Python constants**

```python
FIGURE_FILENAMES = {
    "line_chart": "archetype_share_and_win_rate.png",
    "sankey": "archetype_sankey.png",
    "matchup_matrix": "archetype_matchup_matrix.png",
}
TABLE_FILENAMES = (
    "line_chart_daily_metrics.csv",
    "matchup_matrix.csv",
)
```

Remove filename parsing/validation from `_chart_settings`. Pass explicit staged output paths or chart names into save functions rather than reading filenames from configuration.

- [ ] **Step 5: Stage the entire Deck Trends output set before promotion**

Create a temporary staging directory next to the configured output, generate all three figures and both tables there, validate all five exist, then replace only those five fixed destinations with `os.replace`. Keep all existing analytical preparation and plotting calls unchanged.

- [ ] **Step 6: Run focused Deck Trends tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_deck_trends_eda.py imitation_learning/tests/test_deck_trend.py -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit Deck Trends publication changes**

```powershell
git add imitation_learning/cfg/deck_trends_eda.yaml imitation_learning/notebooks/deck_trends_eda.py imitation_learning/tests/test_deck_trends_eda.py
git commit -m "refactor: fix EDA output filenames"
```

---

### Task 5: Remove the Notebook and Update Documentation

**Files:**
- Delete: `imitation_learning/notebooks/replay_timing.ipynb`
- Modify: `imitation_learning/README.md`
- Modify: `imitation_learning/tests/test_replay_timing_notebook.py`

**Interfaces:**
- Consumes: the two commands completed in Tasks 1-3
- Produces: documented extraction and EDA invocation with config/output locations

- [ ] **Step 1: Update README replay timing instructions**

Replace the notebook paragraph with the two-command workflow:

````markdown
Extract player timing data using `cfg/extract_replay_timing.yaml`:

```powershell
python -m extraction.replay_timing
```

Generate Replay Timing EDA using `cfg/replay_timing_eda.yaml`:

```powershell
python notebooks/replay_timing_eda.py
```

Both configurations accept `date: latest` or an explicit date such as `8.15`.
The EDA writes six fixed figures and three tables under `outputs/replay_timing/`.
````

Also state that successful EDA runs replace fixed outputs while failed runs leave the previous complete results in place.

- [ ] **Step 2: Remove the obsolete notebook**

Run:

```powershell
git rm imitation_learning/notebooks/replay_timing.ipynb
```

Add the final replacement assertion to `test_replay_timing_notebook.py`:

```python
def test_obsolete_replay_timing_notebook_is_removed() -> None:
    assert not (PROJECT_ROOT / "notebooks" / "replay_timing.ipynb").exists()
```

- [ ] **Step 3: Run path and documentation tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_replay_timing_notebook.py imitation_learning/tests/test_artifacts.py -q
```

Expected: the pure-Python replacement assertion passes. If `test_artifacts.py` reports its known unrelated submission-notebook JSON baseline failure, record it separately and confirm no new documentation/path failure exists.

- [ ] **Step 4: Commit notebook removal and docs**

```powershell
git add imitation_learning/README.md imitation_learning/tests/test_replay_timing_notebook.py
git commit -m "docs: document replay timing commands"
```

---

### Task 6: End-to-End Verification

**Files:**
- Modify only if verification exposes a defect in files already listed above.

**Interfaces:**
- Verifies: extraction command, replay timing EDA command, Deck Trends EDA command, and repository tests

- [ ] **Step 1: Run focused tests together**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests/test_replay_timing_extract.py imitation_learning/tests/test_replay_timing_notebook.py imitation_learning/tests/test_deck_trends_eda.py imitation_learning/tests/test_deck_trend.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Smoke-test extraction with available local data**

From `imitation_learning`:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m extraction.replay_timing
```

Expected: prints the resolved date and writes or reuses its dated player-timing CSV. If errors exist, the error CSV row count matches the reported total; otherwise no dated error CSV exists.

- [ ] **Step 3: Smoke-test Replay Timing EDA twice**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' notebooks/replay_timing_eda.py
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' notebooks/replay_timing_eda.py
```

Expected: both runs succeed, print the same resolved date, and the second run replaces all six PNGs and three CSVs without accumulating dated or configurable filenames.

- [ ] **Step 4: Smoke-test Deck Trends EDA**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' notebooks/deck_trends_eda.py
```

Expected: succeeds with the existing three figures and two tables at their fixed paths.

- [ ] **Step 5: Run the full test suite and record baseline failures separately**

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest imitation_learning/tests -q
```

Expected: no new failure from this change. Known unrelated baseline failures or the Windows/PyTorch teardown stall must be reported with exact test names and distinguished from focused-test results.

- [ ] **Step 6: Inspect final repository state**

```powershell
git diff --check
git status --short
git log -6 --oneline
```

Expected: no whitespace errors and no unintended generated outputs staged or tracked.

- [ ] **Step 7: Commit any verification-only fixes**

If verification required source/test corrections, stage only those corrections and commit:

```powershell
git commit -m "fix: complete replay timing EDA migration"
```

If no correction was needed, do not create an empty commit.
