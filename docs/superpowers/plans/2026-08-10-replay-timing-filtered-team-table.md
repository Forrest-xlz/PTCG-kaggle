# Replay Timing Score-Filtered Team Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-row-per-team score and timing table for the active score-filtered cohort in the replay timing EDA notebook.

**Architecture:** Reuse `filtered_teams` for timing and cluster fields, aggregate only `mean_score` from filtered `player_timings`, and merge by `team_name`. Insert one markdown cell and one code cell after cohort summaries without changing extraction, CSV, clustering, or charts.

**Tech Stack:** Jupyter notebook JSON, pandas, pytest.

## Global Constraints

- Preserve all existing notebook cells and outputs except for inserting the new section.
- Do not modify `deck_trend.ipynb`, `deck_trend.yaml`, extraction logic, clustering, charts, or CSV schemas.
- Score remains the active replay-level `score_value`, not an inferred individual-player score.
- Display one unique row per filtered team, sorted by descending mean score.

---

### Task 1: Add and Validate the Filtered Team Detail Table

**Files:**
- Modify: `imitation_learning/eda/replay_timing.ipynb`
- Create: `imitation_learning/tests/test_replay_timing_notebook.py`

**Interfaces:**
- Consumes: `player_timings`, `filtered_mask`, `filtered_teams`, `SCORE_MODE`, and `FILTER_LABEL`.
- Produces notebook variable: `score_filtered_team_details: pd.DataFrame`.

- [ ] **Step 1: Write a failing notebook structure test**

Assert the notebook contains the heading `### Score-Filtered Team Details`, computes `mean_score` from `score_value`, merges with `filtered_teams`, exposes the approved columns, sorts by `mean_score`, and checks team equality.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python -m pytest imitation_learning/tests/test_replay_timing_notebook.py -q`

Expected: FAIL because the section is absent.

- [ ] **Step 3: Insert the markdown and code cells with notebook-safe tooling**

Use `nbformat` to insert the section after the cohort/cluster summary. For a nonempty cohort, group filtered player rows by `team_name`, compute `mean_score`, merge one-to-one with the timing frame, select and rename display columns, validate team identity, sort deterministically, and display rounded values. For an empty cohort, create an empty frame with the same columns and print `No teams satisfy {FILTER_LABEL}.`.

- [ ] **Step 4: Run the focused test**

Run: `python -m pytest imitation_learning/tests/test_replay_timing_notebook.py -q`

Expected: PASS.

- [ ] **Step 5: Validate notebook JSON and execution**

Run JSON/nbformat validation, then execute the notebook top-to-bottom with `nbconvert` if dependencies and replay archives are available. Preserve the executed notebook only if execution succeeds.

- [ ] **Step 6: Commit only the notebook and its focused test**

```bash
git add imitation_learning/eda/replay_timing.ipynb
git add -f imitation_learning/tests/test_replay_timing_notebook.py
git commit -m "feat: show score-filtered replay timing teams"
```
