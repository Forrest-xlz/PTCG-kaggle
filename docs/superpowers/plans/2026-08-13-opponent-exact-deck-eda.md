# Opponent Exact Deck EDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible notebook that selects high-coverage Exact Deck classes from the most recent replay dates and exports their label table.

**Architecture:** The notebook directly reuses `deck.analysis` for validated deck loading, Exact Deck identity, and archetype annotations. It performs bounded pandas aggregation and plotting, then writes only the selected Top K classes into `imitation_learning/data/opponent_deck_classes.csv`.

**Tech Stack:** Python, pandas, matplotlib, nbformat, nbclient/Jupyter.

## Global Constraints

- Exact Deck ID is the classification unit; archetype is explanatory only.
- Top K exclusions receive no `OTHER` class and will later have auxiliary-loss masks disabled.
- Notebook titles and plots use English.
- No model, cache, extraction, or training source changes.

---

### Task 1: Create the analysis notebook

**Files:**
- Create: `imitation_learning/eda/opponent_deck_classification.ipynb`

**Interfaces:**
- Consumes: `imitation_learning/data/deck/*.decks.csv`, the configured card table, and `deck.analysis`.
- Produces: reader-facing tables/charts and `imitation_learning/data/opponent_deck_classes.csv`.

- [ ] Add setup and a single parameter cell with `RECENT_DATE_COUNT`, `MIN_UNIQUE_REPLAYS`, `TOP_K`, and `OUTPUT_NAME`.
- [ ] Discover and calendar-sort source dates, select the latest N, load and annotate rows.
- [ ] Aggregate stable Exact Deck statistics and calculate valid auxiliary-label coverage from the opponent perspective.
- [ ] Display eligible and Top K tables, coverage summary, bar chart, and daily trend chart.
- [ ] Export Top K with stable zero-based `class_id`, 60-card JSON, and explanatory archetype fields.

### Task 2: Execute and validate the artifact

**Files:**
- Validate: `imitation_learning/eda/opponent_deck_classification.ipynb`
- Create: `imitation_learning/data/opponent_deck_classes.csv`

**Interfaces:**
- Confirms the notebook runs top-to-bottom against the current project data.

- [ ] Validate notebook JSON and run every cell with the project Python environment.
- [ ] Confirm the CSV row count is `min(TOP_K, eligible_count)`, class IDs are contiguous, Deck IDs are unique, and every deck contains 60 cards.
- [ ] Inspect outputs for bounded tables and readable chart titles.
- [ ] Run `git diff --check` and commit the notebook; the ignored data CSV remains a generated local artifact.
