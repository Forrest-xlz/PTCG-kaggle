# Force-Include Date for Full Training

## Goal

Keep the `ver_final_0` full-training pipeline unchanged except for one data
eligibility rule: every cached player sample from `8.16` participates in
training, regardless of its recorded win/loss/draw result. Other dates remain
winner-only.

## Configuration

Add one explicit training setting:

```yaml
include_all_dates: ["8.16"]
```

Remove the loser-augmentation section from `train.yaml`. The training entry
point no longer loads score manifests or selects expert losing episodes.

## Data Flow

Extraction and caching remain unchanged at schema versions 4 and 16. During
training-index construction:

- a shard whose parsed date is in `include_all_dates` uses every sample;
- every other shard uses only samples whose `player_result` is `win`;
- replay-level `train_replay_ratio` sampling remains unchanged.

Because the training entry point does not load score manifests, `8.16` does
not need a `manifest.csv`. Existing extracted data and cache shards remain
compatible. If `8.16` is new, running extract and cache with `force: false`
only processes the new source.

## Compatibility and Scope

- Do not change model architecture, features, extraction schema, or cache
  schema.
- Do not add date or team oversampling.
- Preserve lower-level loser-selection support in `feature_cache.py` where
  removing it would create unrelated churn, but do not call it from full
  training.
- Preserve the user's unrelated modified `eda/deck_trend.ipynb`.

## Validation

- A cache fixture proves win/loss/draw samples are all selected on `8.16`.
- A neighboring date remains winner-only.
- An absent configured date fails clearly.
- Training configuration parses `include_all_dates` and no longer requires
  loser-augmentation settings.
- Targeted tests and Python compilation complete successfully.
