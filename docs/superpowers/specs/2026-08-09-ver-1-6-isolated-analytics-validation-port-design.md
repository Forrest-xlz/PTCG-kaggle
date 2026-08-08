# Ver 1.6 Isolated Analytics and Validation Port

## Goal

Keep `ver_1.6.0` based on `ver_1.4.5` and add only:

1. ordered per-deck top-deck validation;
2. deck trend extraction and EDA;
3. replay timing EDA.

The model architecture, feature representation, cache schema, action-history
pipeline, submission agent, and existing model YAML settings must remain
unchanged.

## Validation behavior

`train.top_decks` remains an ordered list of exact 60-card decks. Its order
defines `deck1`, `deck2`, and so on. Latest-date and in-distribution validation
each run the model once and calculate subgroup metrics from masks over the same
logits.

The existing overall groups remain:

- `val_in_distribution/*`
- `val_in_distribution_expert/*`
- `val_latest/*`
- `val_latest_expert/*`

Each configured deck adds four independent WandB groups:

- `val_in_distribution_deckN/*`
- `val_in_distribution_expert_deckN/*`
- `val_latest_deckN/*`
- `val_latest_expert_deckN/*`

Each validation group contains only cross-entropy loss and top-1, top-3, and
top-5 accuracy. Sample counts remain data metadata rather than validation
curves. Empty configured deck subsets fail before training with a descriptive
error. Duplicate exact decks are rejected.

The split implementation may add runtime masks to `DatasetSplits`, but it must
not change packed-cache fields or increment the cache schema version. Existing
caches remain usable.

## Deck trend EDA

Port the isolated deck-trend workflow:

- `imitation_learning/cfg/deck_trend.yaml`
- `imitation_learning/deck/trend.py`
- `imitation_learning/deck/trend_extract.py`
- `imitation_learning/eda/deck_trend.ipynb`

The extractor saves reusable deck-level data independently of archetype
classification. The notebook applies the current manual archetype rules at
analysis time, groups dates using the configurable interval, maps decks beyond
the configured cutoff to `Other`, and visualizes usage, win-rate, and matchup
trends. Changing classification or interval settings does not require replay
extraction again.

## Replay timing EDA

Port only the established replay timing notebook to
`imitation_learning/eda/replay_timing.ipynb`, plus its direct dependency and
README entries. Do not copy training, feature, model, cache, submission, or
configuration files from the older branch or archive.

The notebook keeps its existing behavior: latest-date replay analysis,
team-level aggregation, configurable score threshold and `avg`/`min`/`max`
score mode, startup/subsequent-action distributions, globally fitted four-way
K-means clusters, and CSV output under `imitation_learning/data`.

## Isolation safeguards

Implementation uses selective file/hunk transfer rather than merging the old
branch or copying the complete archive. The following files must have no diff
from the `ver_1.4.5` baseline:

- `imitation_learning/model/features.py`
- `imitation_learning/model/network.py`
- `imitation_learning/training/cache_features.py`
- `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- `imitation_learning/cfg/train.yaml`

Within `training/train.py`, existing `ver_1.4.5` model settings and their
transfer to `ModelConfig` must remain intact. Within `training/feature_cache.py`,
history arrays, cached layouts, schema constants, and collation remain intact.

## Verification

Verification includes focused tests for ordered per-deck masks, expert
intersections, shared validation forwards, WandB namespace contents, deck trend
extraction/analysis, and notebook structure. A final baseline diff checks that
the protected model and pipeline files are unchanged. If Python is unavailable
locally, static Git checks are reported separately and executable tests are not
claimed.
