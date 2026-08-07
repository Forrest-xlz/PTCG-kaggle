# Replay Timing EDA Migration Design

## Goal

Restore the already-adjusted replay timing EDA from `ver_1.3.1.zip` into the
current project while leaving deck, model, training, cache, and submission
logic unchanged.

## Source of Truth

The source artifact is exactly:

```text
D:\learning\AI\Kaggle\ver_1.3.1.zip
ZIP entry: imitation_learning/eda/replay_timing.ipynb
```

The notebook is copied without changing its code cells, parameter defaults,
timing formulas, score filtering, aggregation, clustering, figures, output
paths, metadata, or stored outputs. Its logic is treated as already reviewed
and must not be reimplemented or reformatted.

## Destination

Create only:

```text
imitation_learning/eda/replay_timing.ipynb
```

Do not migrate the old `eda/deck.ipynb`. Do not move or edit files under
`imitation_learning/deck/`, `model/`, `training/`, or `cfg/`.

## Environment and Documentation

Add the notebook dependencies that are absent from the current
`imitation_learning/requirements.txt`:

```text
seaborn>=0.12
scikit-learn>=1.3
jupyter>=1.0
nbformat>=5.9
```

Add a concise README entry explaining that
`eda/replay_timing.ipynb` analyzes the newest dated replay ZIP and writes
cached player/team tables under `data/replay_timing/`. Do not duplicate the
notebook's full analytical method in README.

## Validation

- Compare the migrated notebook bytes with the ZIP entry using SHA-256.
- Parse the destination as notebook JSON and require at least one cell.
- Confirm all code-cell sources are identical to the ZIP source.
- Confirm no files outside the notebook, requirements, README, design, and
  implementation plan are modified.
- Execute the notebook top-to-bottom only when a Python/Jupyter environment is
  available. If unavailable, report the exact execution command and validation
  gap instead of changing notebook logic.
