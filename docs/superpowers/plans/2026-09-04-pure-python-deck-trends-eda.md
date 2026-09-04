# Pure-Python Deck Trends EDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Deck Trends notebook with one fixed-YAML Python EDA command that creates three independently configured figures and two audit tables.

**Architecture:** Split extraction and analysis YAMLs, add reusable score/config/plot helpers to `analysis.deck_trends`, and keep orchestration in `notebooks/deck_trends_eda.py`. Each chart independently selects dates and replays, then writes one stable output.

**Tech Stack:** Python 3.10+, pandas, NumPy, Matplotlib, Seaborn, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-09-04-pure-python-deck-trends-eda-design.md`

## Global Constraints

- Work on branch `end`; do not merge or push without user direction.
- The approved migration deletes the locally executed `notebooks/deck_trends.ipynb`.
- Do not modify other EDA notebooks, deck classification, extraction schemas, training, or models.
- Keep fixed YAML paths; do not add CLI configuration arguments.
- Use inclusive `>=` for `min`, `max`, and `avg` score thresholds.
- Use `C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe` for tests.

---

### Task 1: Split extraction and EDA configuration

**Files:**
- Move: `imitation_learning/cfg/analyze_deck_trends.yaml` -> `imitation_learning/cfg/extract_deck_trend_data.yaml`
- Create: `imitation_learning/cfg/deck_trends_eda.yaml`
- Modify: `imitation_learning/extraction/deck_trend_data.py`
- Test: `imitation_learning/tests/test_deck_trend_extract.py`

**Interfaces:**
- Produces extraction `CONFIG_PATH` ending in `extract_deck_trend_data.yaml`.
- Produces EDA sections `analysis.line_chart`, `analysis.sankey`, and `analysis.matchup_matrix` exactly as specified.

- [ ] Change the extraction config-path test to require the new filename and add a YAML test that the extraction file contains only `extract` while the EDA file contains only `analysis`.
- [ ] Run the focused tests and verify they fail because the new YAMLs do not exist.
- [ ] Move the YAML, remove its `analysis` section, create the EDA YAML from the approved spec, and update `extraction/deck_trend_data.py`.
- [ ] Run the focused tests and verify they pass.
- [ ] Commit as `refactor: separate deck trend configurations`.

### Task 2: Add reusable score and chart-data helpers

**Files:**
- Modify: `imitation_learning/analysis/deck_trends.py`
- Modify: `imitation_learning/tests/test_deck_trend.py`

**Interfaces:**
- Produces `ScoreFilter(mode: str, threshold: float | None)`.
- Produces `filter_replays_by_score(rows, scores, score_filter) -> pd.DataFrame`.
- Produces `select_chart_rows(rows, interval_days, start_date, end_date) -> tuple[pd.DataFrame, list[str]]`.
- Produces `plot_share_and_win_rate(metrics, dates, min_share_percent) -> Figure`.
- Produces `plot_matchup_matrix(matchups, archetypes, title) -> Figure`.

- [ ] Add table-driven failing tests for `all`, `min`, `max`, and `avg`, including equality at the threshold and invalid mode/threshold combinations.
- [ ] Add failing tests proving chart date selection is independent and matrix archetypes use pooled share after score filtering.
- [ ] Implement the smallest validated dataclass and helpers, reusing existing parsing, daily metrics, pooled shares, and matchup calculations.
- [ ] Rename the daily metric output field to `win_rate` while preserving mirror-filter behavior and update focused expectations.
- [ ] Run `tests/test_deck_trend.py` and verify no new failures beyond the three recorded baseline failures.
- [ ] Commit as `feat: add reusable deck trend EDA helpers`.

### Task 3: Replace the notebook with the pure-Python EDA

**Files:**
- Delete: `imitation_learning/notebooks/deck_trends.ipynb`
- Create: `imitation_learning/notebooks/deck_trends_eda.py`
- Replace: `imitation_learning/tests/test_deck_trend_notebook.py` -> `imitation_learning/tests/test_deck_trends_eda.py`

**Interfaces:**
- Consumes `cfg/deck_trends_eda.yaml` and the Task 2 helpers.
- Produces exactly three configured PNG files plus `tables/line_chart_daily_metrics.csv` and `tables/matchup_matrix.csv`.

- [ ] Write a failing controlled-fixture test that invokes `main(config_path)` and asserts three non-empty PNGs, two exact CSV paths, hidden low-share rows in the line table, and no `selected_dates.csv`.
- [ ] Implement configuration dataclasses/load validation, trend-shard loading, shared manifest loading, per-chart selection, plot saving, CSV writing, logging, and explicit figure closing.
- [ ] Ensure `main()` defaults to `cfg/deck_trends_eda.yaml`, while the optional `config_path` argument exists only for tests—not as a CLI flag.
- [ ] Remove the old Notebook after the script behavior passes.
- [ ] Run the new EDA tests and existing extraction/trend tests.
- [ ] Commit as `refactor: replace deck trend notebook with Python EDA`.

### Task 4: Documentation and final verification

**Files:**
- Modify: `imitation_learning/README.md`
- Modify: tracked artifact/layout tests that reference the old Notebook or YAML.

**Interfaces:**
- Produces documented commands `python -m extraction.deck_trend_data` and `python notebooks/deck_trends_eda.py`.

- [ ] Update README paths, explain chart-specific parameters and score modes, and document the five outputs.
- [ ] Search tracked sources for `analyze_deck_trends.yaml` and `deck_trends.ipynb`; fix every non-historical reference.
- [ ] Parse both YAMLs, compile all modified Python files without writing `.pyc`, and run focused tests.
- [ ] Run the full suite; report existing baseline failures separately from regressions.
- [ ] Run `git diff --check`, verify only intended files changed, and commit as `docs: document pure Python deck trends EDA`.
- [ ] Invoke `superpowers:verification-before-completion` and report exact evidence.
