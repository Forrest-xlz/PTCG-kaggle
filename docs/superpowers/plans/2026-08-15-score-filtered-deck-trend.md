# Score-Filtered Deck Trend Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently configurable mean-score-filtered Sankey and matchup-matrix cells to the existing Deck Trend notebook.

**Architecture:** The first new cell reads `manifest.csv` from each selected replay ZIP, joins `avg_score` to the already classified replay rows, and builds a filtered Sankey using the existing trend helpers. The second new cell reuses those scored rows and builds a filtered matchup matrix with its own threshold, leaving extraction and all original notebook outputs unchanged.

**Tech Stack:** Python, Jupyter/nbformat, pandas, matplotlib, seaborn, `zipfile`, existing `deck.trend` helpers.

## Global Constraints

- Do not modify `deck/trend_extract.py` or extracted shard schemas.
- Do not alter existing charts, exports, YAML values, or outputs.
- Use strict `avg_score > threshold` filtering based on the replay-level two-player mean in `manifest.csv`.
- Keep `SANKEY_MIN_AVG_SCORE` and `MATCHUP_MIN_AVG_SCORE` local to their respective cells.
- Continue honoring `snapshot_dates`, `MIN_SHARE_PERCENT`, and `EXCLUDE_MIRROR_MATCHES`.
- Preserve existing uncommitted notebook and YAML changes.

---

### Task 1: Add the score-filtered Sankey view

**Files:**
- Modify: `imitation_learning/eda/deck_trend.ipynb`

**Interfaces:**
- Consumes: `extract_cfg`, `PROJECT_ROOT`, `snapshot_dates`, `annotated`, `MIN_SHARE_PERCENT`, `analysis_dir`, and existing trend helper imports.
- Produces: `annotated_with_scores: pandas.DataFrame` with numeric `avg_score`, plus the filtered Sankey figure and PNG.

- [ ] **Step 1: Run a structural assertion that fails before implementation**

```powershell
& $python -c "import json; nb=json.load(open(r'imitation_learning/eda/deck_trend.ipynb',encoding='utf-8')); text='\n'.join(''.join(c.get('source',[])) for c in nb['cells']); assert 'SANKEY_MIN_AVG_SCORE' in text"
```

Expected: `AssertionError` because the new cell does not yet exist.

- [ ] **Step 2: Insert one code cell directly after the existing Sankey code cell**

The cell must:

```python
SANKEY_MIN_AVG_SCORE = 1100.0

import io
import zipfile

replay_input_dir = Path(extract_cfg['input'])
if not replay_input_dir.is_absolute():
    replay_input_dir = PROJECT_ROOT / replay_input_dir

manifest_frames = []
for date_label in snapshot_dates:
    archive_path = replay_input_dir / f'{date_label}.zip'
    if not archive_path.is_file():
        raise FileNotFoundError(f'Replay archive not found: {archive_path}')
    with zipfile.ZipFile(archive_path) as replay_zip:
        with replay_zip.open('manifest.csv') as raw:
            manifest = pd.read_csv(
                io.TextIOWrapper(raw, encoding='utf-8'),
                dtype={'episode_id': str},
                usecols=['episode_id', 'avg_score'],
            )
    manifest.insert(0, 'date', str(date_label))
    manifest_frames.append(manifest)

replay_scores = pd.concat(manifest_frames, ignore_index=True)
replay_scores['avg_score'] = pd.to_numeric(
    replay_scores['avg_score'], errors='raise'
)
if not np.isfinite(replay_scores['avg_score']).all():
    raise ValueError('manifest avg_score must be finite')
if replay_scores.duplicated(['date', 'episode_id']).any():
    raise ValueError('manifest contains duplicate date/episode_id rows')

annotated_with_scores = annotated.merge(
    replay_scores,
    on=['date', 'episode_id'],
    how='left',
    validate='many_to_one',
)
if annotated_with_scores['avg_score'].isna().any():
    missing = annotated_with_scores.loc[
        annotated_with_scores['avg_score'].isna(), ['date', 'episode_id']
    ].drop_duplicates()
    raise ValueError(f'Missing manifest scores for {len(missing):,} replay(s)')

score_sankey_rows = annotated_with_scores[
    annotated_with_scores['avg_score'] > SANKEY_MIN_AVG_SCORE
].copy()
if score_sankey_rows.empty:
    raise ValueError(
        f'No replay has avg_score > {SANKEY_MIN_AVG_SCORE:g}'
    )
score_pair_sizes = score_sankey_rows.groupby(['date', 'episode_id']).size()
assert score_pair_sizes.eq(2).all()
```

It then recomputes filtered daily metrics, daily share lookup, team modal archetypes, display shares, display flows, and calls `plot_archetype_sankey()` using the same logic as the original cell. Save the chart as:

```python
score_sankey_path = analysis_dir / (
    f'archetype_sankey_avg_score_gt_{SANKEY_MIN_AVG_SCORE:g}.png'
)
```

Print total selected replay count and filtered replay count.

- [ ] **Step 3: Validate the notebook structure**

```powershell
& $python -c "import nbformat; p=r'imitation_learning/eda/deck_trend.ipynb'; nb=nbformat.read(p,as_version=4); nbformat.validate(nb); text='\n'.join(c.source for c in nb.cells); assert text.count('SANKEY_MIN_AVG_SCORE') >= 2"
```

Expected: exit code `0`.

### Task 2: Add the score-filtered matchup matrix

**Files:**
- Modify: `imitation_learning/eda/deck_trend.ipynb`

**Interfaces:**
- Consumes: `annotated_with_scores`, `global_visible_archetypes`, `EXCLUDE_MIRROR_MATCHES`, `analysis_dir`, and `build_matchups()`.
- Produces: filtered heatmap and `matchup_matrix_avg_score_gt_<threshold>_long.csv`.

- [ ] **Step 1: Run a structural assertion that fails before implementation**

```powershell
& $python -c "import json; nb=json.load(open(r'imitation_learning/eda/deck_trend.ipynb',encoding='utf-8')); text='\n'.join(''.join(c.get('source',[])) for c in nb['cells']); assert 'MATCHUP_MIN_AVG_SCORE' in text"
```

Expected: `AssertionError` because the second new cell does not yet exist.

- [ ] **Step 2: Insert one code cell directly after the existing matchup code cell**

The cell starts with:

```python
MATCHUP_MIN_AVG_SCORE = 1100.0

score_matchup_rows = annotated_with_scores[
    annotated_with_scores['avg_score'] > MATCHUP_MIN_AVG_SCORE
].copy()
if score_matchup_rows.empty:
    raise ValueError(
        f'No replay has avg_score > {MATCHUP_MIN_AVG_SCORE:g}'
    )
score_pair_sizes = score_matchup_rows.groupby(['date', 'episode_id']).size()
assert score_pair_sizes.eq(2).all()

score_matchups = build_matchups(
    score_matchup_rows,
    exclude_mirrors=EXCLUDE_MIRROR_MATCHES,
)
assert (score_matchups['wins'] + score_matchups['losses'] + score_matchups['draws']).equals(
    score_matchups['games']
)
assert score_matchups['win_rate'].dropna().between(0, 1).all()
```

Use `global_visible_archetypes` for both axes and retain the original heatmap annotations, color scale, and row-beats-column orientation. Include `avg_score > threshold` in the title and save the long table to:

```python
score_matchup_path = analysis_dir / (
    f'matchup_matrix_avg_score_gt_{MATCHUP_MIN_AVG_SCORE:g}_long.csv'
)
```

Print total selected replay count, filtered replay count, and saved path.

- [ ] **Step 3: Validate notebook syntax and cell placement**

```powershell
& $python -c "import ast,nbformat; p=r'imitation_learning/eda/deck_trend.ipynb'; nb=nbformat.read(p,as_version=4); nbformat.validate(nb); cells=[c.source for c in nb.cells]; si=next(i for i,s in enumerate(cells) if 'SANKEY_MIN_AVG_SCORE' in s); mi=next(i for i,s in enumerate(cells) if 'MATCHUP_MIN_AVG_SCORE' in s); assert si < mi; [ast.parse(s) for s in cells if not s.lstrip().startswith('%')]"
```

Expected: exit code `0`.

### Task 3: Run score-join and filtered-analysis smoke validation

**Files:**
- Validate: `imitation_learning/eda/deck_trend.ipynb`

**Interfaces:**
- Consumes: the completed notebook and locally configured replay/deck-trend data.
- Produces: evidence that manifests join completely and both score-filtered datasets preserve two player rows per replay.

- [ ] **Step 1: Execute notebook cells through the filtered matchup cell in a temporary kernel**

Use `nbclient` or an equivalent temporary executed copy so source notebook outputs are not rewritten. Skip only the existing `%pip install` cell if the local environment already has a valid `nbformat` version.

- [ ] **Step 2: Confirm runtime invariants**

Expected:

```text
missing manifest scores = 0
rows per filtered replay = 2
filtered Sankey replays > 0
filtered matchup replays > 0
all matchup win rates are within [0, 1]
```

- [ ] **Step 3: Inspect the final diff**

```powershell
git diff --stat -- imitation_learning/eda/deck_trend.ipynb
git diff -- imitation_learning/eda/deck_trend.ipynb
```

Confirm that the notebook contains exactly two added code cells and no changes to existing cell sources.
