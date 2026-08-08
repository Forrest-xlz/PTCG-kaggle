# Deck Trend EDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, cached Deck-trend extraction and notebook workflow with configurable calendar-day snapshots, curated-rule-only archetypes, team switching, win-rate trends, Sankey flows, and matchup analysis.

**Architecture:** `deck/trend_extract.py` performs the expensive replay ZIP scan once and writes schema-versioned per-date player shards containing complete Deck lists and team identity. `deck/trend.py` contains pure classification and analysis helpers so both tests and the notebook use one implementation. `eda/deck_trend.ipynb` reads YAML and cached shards only, then renders and exports the analysis.

**Tech Stack:** Python 3.10+, pandas, PyYAML, Plotly, matplotlib/seaborn, pytest, Jupyter/nbformat.

## Global Constraints

- Baseline classification uses ordered `deck.analysis.ARCHETYPE_RULES` only; every unmatched Deck is `Other`.
- Do not change `deck.analysis.classify_deck()` or any training, model, cache, validation, or submission behavior.
- Extraction output must retain all 60 Card IDs and must not persist an archetype label.
- `snapshot_interval_days` is a positive calendar-day gap, not a row/index stride.
- Changing classifier or analysis settings must not require re-extraction.
- Per-date extraction is resumable; `force: false` reuses only a current-schema, complete shard.

---

### Task 1: Pure Trend Classification and Analysis Helpers

**Files:**
- Create: `imitation_learning/deck/trend.py`
- Create: `imitation_learning/tests/test_deck_trend.py`

**Interfaces:**
- Consumes: `deck.analysis.ARCHETYPE_RULES`, `CardCatalog`, `normalize_card_name`, and valid sorted 60-card lists.
- Produces: `classify_trend_archetype(deck, catalog) -> str`, `parse_month_day(value) -> date`, `select_snapshot_dates(values, interval_days, start_date=None, end_date=None) -> list[str]`, `add_trend_archetypes(rows, catalog) -> DataFrame`, `build_daily_metrics(rows, exclude_mirrors) -> DataFrame`, `build_team_modal_archetypes(rows) -> DataFrame`, `build_team_flows(modal, dates) -> tuple[DataFrame, DataFrame]`, and `build_matchups(rows, exclude_mirrors) -> DataFrame`.

- [ ] **Step 1: Write focused failing tests**

Create fixtures with a tiny `CardCatalog` and synthetic paired player rows. Assert:

```python
assert classify_trend_archetype(grimmsnarl_deck, catalog) == "Marnie Grimmsnarl"
assert classify_trend_archetype(unmatched_deck, catalog) == "Other"
assert select_snapshot_dates(
    ["6.26", "6.27", "6.29", "7.1", "7.2"], 3
) == ["6.26", "6.29", "7.2"]
```

Also assert deterministic alphabetical modal tie-breaking, transition counts for two adjacent snapshots, per-date share summing to `100.0`, mirror exclusion, and matchup win/loss reconciliation.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
python -m pytest imitation_learning/tests/test_deck_trend.py -q
```

Expected: import failure because `deck.trend` does not exist.

- [ ] **Step 3: Implement minimal pure helpers**

Implement name-set matching in the same ordered `all`/`any` form used by the current classifier, returning only the rule name or `Other`. Parse `month.day` on a fixed leap-safe reference year and return canonical unpadded `month.day` labels. Select the earliest eligible date and then the earliest date whose calendar distance from the preceding selection is at least `interval_days`.

For paired rows, derive `opponent_archetype` by joining on `date`, `episode_id`, and the opposite player index. Daily metrics must expose `date`, `archetype`, `uses`, `share_percent`, `wins`, `losses`, `draws`, `non_mirror_games`, and `non_mirror_win_rate`. Flow output must expose `from_date`, `to_date`, `source_archetype`, `target_archetype`, and `teams`; retention output must expose both date team counts, overlap, unchanged, switched, and switch rate.

- [ ] **Step 4: Run the focused test and verify pass**

Run the same pytest command and expect all tests in the file to pass.

- [ ] **Step 5: Commit the independently testable helper layer**

```bash
git add imitation_learning/deck/trend.py imitation_learning/tests/test_deck_trend.py
git commit -m "feat: add deck trend analysis helpers"
```

---

### Task 2: Resumable Replay Trend Extraction

**Files:**
- Create: `imitation_learning/deck/trend_extract.py`
- Create: `imitation_learning/cfg/deck_trend.yaml`
- Create: `imitation_learning/tests/test_deck_trend_extract.py`

**Interfaces:**
- Consumes: replay ZIP files, `deck.extract.extract_decks`, `cfg/deck_trend.yaml`.
- Produces: `<date>.players.csv.gz`, `<date>.meta.json`, `TrendExtractSettings`, `load_settings()`, `extract_team_names(payload)`, and `process_archive(args)`.

- [ ] **Step 1: Write extractor tests with a temporary ZIP**

Build a minimal replay payload containing `info.EpisodeId`, `info.TeamNames`, two 60-card lists under initial visualization actions, and final rewards. Assert the produced gzip CSV has exactly two rows and columns:

```python
[
    "date", "episode_id", "player", "team_name",
    "opponent_team_name", "deck", "reward", "result",
]
```

Assert team-name fallback to `info.Agents[*].Name`, invalid names/decks become bounded meta errors, limited extraction records `complete: false`, full extraction records `complete: true`, and `force: false` skips only a current-schema complete shard.

- [ ] **Step 2: Run the extractor test and verify failure**

```bash
python -m pytest imitation_learning/tests/test_deck_trend_extract.py -q
```

Expected: import failure because `deck.trend_extract` does not exist.

- [ ] **Step 3: Implement YAML validation and atomic gzip output**

Define `TrendExtractSettings(input, output, workers, limit_members, force)` and validate exact types and ranges. Use per-archive worker processes. Each worker sorts the Deck list, serializes compact JSON, derives result from reward, writes UTF-8 gzip CSV through a temporary file followed by `os.replace`, and atomically writes metadata with schema version, archive name, archive size/mtime, processed games, player rows, failures, bounded errors, limit, and completeness.

- [ ] **Step 4: Implement incremental main flow**

Discover ZIP files, cap worker count to archive count, submit one archive per process, and print one compact JSON summary per completed archive. Never apply `analysis.snapshot_interval_days` during extraction.

- [ ] **Step 5: Run extraction tests and the existing Deck extractor tests**

```bash
python -m pytest imitation_learning/tests/test_deck_trend_extract.py imitation_learning/tests/test_deck_extract.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit extraction and configuration**

```bash
git add imitation_learning/deck/trend_extract.py imitation_learning/cfg/deck_trend.yaml imitation_learning/tests/test_deck_trend_extract.py
git commit -m "feat: add cached deck trend extraction"
```

---

### Task 3: Reader-Facing Deck Trend Notebook

**Files:**
- Create: `imitation_learning/eda/deck_trend.ipynb`
- Modify: `imitation_learning/requirements.txt`
- Modify: `imitation_learning/README.md`

**Interfaces:**
- Consumes: `cfg/deck_trend.yaml`, `data/deck_trend/*.players.csv.gz`, `pokemon_tcg_ai_battle/EN_Card_Data.csv`, and pure helpers from `deck.trend`.
- Produces: charts plus `snapshot_dates.csv`, `archetype_daily_metrics.csv`, `team_modal_archetypes.csv`, `team_snapshot_flows.csv`, `team_retention.csv`, and `matchup_matrix_long.csv` under `data/deck_trend/analysis/`.

- [ ] **Step 1: Add the notebook dependency**

Add `plotly>=5.18` to `imitation_learning/requirements.txt`; existing pandas, PyYAML, seaborn, Jupyter, and nbformat dependencies are reused.

- [ ] **Step 2: Generate a valid analysis notebook with nbformat**

Create a top-to-bottom notebook with sections:

1. `## Goal`
2. `## Setup and Configuration`
3. `## Load and Validate Extracted Data`
4. `## Select Date Snapshots`
5. `## Classification Coverage`
6. `## Archetype Usage and Win Rate`
7. `## Team Modal Archetypes and Retention`
8. `## Sankey Flow`
9. `## Matchup Matrix`
10. `## Exported Tables`
11. `## Takeaways`

The setup cell finds the project root robustly, loads YAML, and displays selected parameters. Data loading reads gzip shards only. Classification deduplicates the serialized Deck strings before calling `classify_trend_archetype` and maps results back.

- [ ] **Step 3: Add bounded visuals**

Use line charts for share and non-mirror win rate, a Plotly Sankey whose low-share nodes are display-grouped into `Other`, a retention table, and a seaborn matchup heatmap annotated with win rate and game count. Keep chart titles and axes in English to avoid font issues.

- [ ] **Step 4: Add data-quality assertions and exports**

Assert two rows per replay, player indices `{0, 1}`, 60-card Decks, nonempty team names, shares summing to 100 percent, and valid win-rate bounds. Export the six specified tables with UTF-8 BOM CSV encoding.

- [ ] **Step 5: Document the workflow**

Add concise README commands:

```bash
python imitation_learning/deck/trend_extract.py
python -m jupyter notebook imitation_learning/eda/deck_trend.ipynb
```

Explain that interval/classifier changes require only rerunning the notebook, while new replay archives require rerunning the incremental extractor first.

- [ ] **Step 6: Validate notebook structure**

Parse with `nbformat.read(..., as_version=4)`, verify required section titles and symbols, and run `nbformat.validate()`.

- [ ] **Step 7: Execute a smoke notebook when Python/Jupyter is available**

Point a temporary config/output at a one-member extraction, execute a temporary notebook copy using:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace <temporary-notebook> --ExecutePreprocessor.timeout=-1
```

Expected: exit code 0 and all six analysis CSVs exist. Do not overwrite the source notebook with machine-specific outputs.

- [ ] **Step 8: Commit notebook and documentation**

```bash
git add imitation_learning/eda/deck_trend.ipynb imitation_learning/requirements.txt imitation_learning/README.md
git commit -m "feat: add deck trend EDA notebook"
```

---

### Task 4: Final Regression and Scope Verification

**Files:**
- Verify all files from Tasks 1-3.
- Add: `docs/superpowers/plans/2026-08-09-deck-trend-eda-implementation.md` to the final commit if still untracked/ignored.

**Interfaces:**
- Consumes: the completed extraction, helper, notebook, tests, and documentation.
- Produces: verification evidence and a clean, scoped branch.

- [ ] **Step 1: Run targeted tests**

```bash
python -m pytest imitation_learning/tests/test_deck_trend.py imitation_learning/tests/test_deck_trend_extract.py -q
```

- [ ] **Step 2: Run the full existing suite**

```bash
python -m pytest imitation_learning/tests -q
```

- [ ] **Step 3: Check formatting and scope**

```bash
git diff --check
git status --short
git diff --stat HEAD~3..HEAD
```

Confirm no model, training, validation, cache, submission, or existing Deck EDA file changed.

- [ ] **Step 4: Commit the implementation plan if necessary**

```bash
git add -f docs/superpowers/plans/2026-08-09-deck-trend-eda-implementation.md
git commit -m "docs: add deck trend implementation plan"
```

- [ ] **Step 5: Report runtime gaps honestly**

If Python/Jupyter is unavailable, report that tests and notebook execution were not run, list the completed static checks, and provide the exact commands above for the user's environment.
