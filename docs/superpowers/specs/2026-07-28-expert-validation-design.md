# Expert Validation Subsets Design

## Goal

Add daily score-ranked expert subsets to the existing in-distribution and
latest-date validation sets. Expert metrics must reuse the parent validation
forward pass rather than evaluating the same samples a second time.

## Manifest Source

Every replay archive under `replay_episodes` contains one `manifest.csv` with:

```text
episode_id,create_time,avg_score,min_score,sum_score,agent_count,size_bytes
```

Training reads manifests directly from the ZIP archives. No extracted manifest
directory or generated intermediate CSV is required.

`train.yaml` adds:

```yaml
train:
  replay_episodes: ../replay_episodes
  expert_validation_ratio: 0.05
```

`replay_episodes` resolves relative to the `imitation_learning` project root.
`expert_validation_ratio` must be strictly between zero and one.

## Daily Score Ranking

Expert thresholds are calculated independently for each replay date. For a
two-player manifest row:

```text
lower_score = min_score
higher_score = sum_score - min_score
```

The ranking population contains both player scores from every valid episode,
so a date with `N` episodes contributes `2N` scores.

For ratio `r`:

```text
top_count = max(1, ceil(2N * r))
cutoff = the top_count-th score after descending sort
```

An episode qualifies as an expert episode when:

```text
higher_score >= cutoff
```

All episodes tied at the cutoff are retained. Therefore, the percentage of
qualifying episodes is intentionally not fixed at `r`; `r` describes the
top-player score population.

The manifest does not identify whether the higher score belongs to player 0
or player 1. Existing replay extraction stores only winning-side decisions.
Consequently, an expert subset means the currently cached winning-side samples
from a qualifying high-score episode. It does not claim that each retained
action was made by the higher-scored player.

## Manifest Validation

Training fails before model construction when any required archive or manifest
is unusable:

- no ZIP matches a cache source date;
- ZIP has no `manifest.csv` or has more than one ambiguous manifest;
- required columns are absent;
- episode ID is empty;
- score fields are non-numeric or non-finite;
- `agent_count` is not exactly two;
- the same episode appears twice in one manifest;
- a parent validation set produces an empty expert subset.

Each cache source date must have a corresponding manifest. Extra replay
archives that do not appear in the current cache are ignored.

At startup, training prints each used date's player-score cutoff, manifest
episode count, and qualifying expert episode count.

## Mapping Manifests to Cached Samples

The feature cache already stores a stable `uint32 episode_key` for every
sample. Manifest episode IDs are converted with the same
`stable_episode_key()` function.

For each cache shard, training uses its parsed source date to select that
date's expert episode-key set. Existing split construction still produces:

```text
train
in_distribution
latest
```

It additionally returns Boolean masks aligned with the two validation-index
arrays:

```text
in_distribution_expert_mask
latest_expert_mask
```

The masks identify expert samples inside their parent validation sets. Expert
samples never form independent datasets and do not affect train/validation
membership.

A 32-bit hash collision can theoretically assign an unrelated episode to an
expert mask. This risk is accepted because the existing validation grouping
already uses the same compact episode key and the collision probability is
negligible for the expected episode count.

## Single-Pass Validation Metrics

Every `eval_every_steps` trigger still performs exactly two validation
traversals:

1. `val_in_distribution`
2. `val_latest`

For each validation batch, the model produces logits once. The evaluator
updates the full-set accumulator with all rows and the expert accumulator with
rows selected by the aligned expert-mask slice:

```python
logits = model(batch)
full_metrics.update(logits, targets, action_counts)
expert_metrics.update(
    logits[expert_mask],
    targets[expert_mask],
    action_counts[expert_mask],
)
```

This yields four metric namespaces from two model traversals:

```text
val_in_distribution/loss
val_in_distribution/top1_accuracy
val_in_distribution/top3_accuracy
val_in_distribution/top5_accuracy
val_in_distribution/samples
val_in_distribution/seconds

val_in_distribution_expert/loss
val_in_distribution_expert/top1_accuracy
val_in_distribution_expert/top3_accuracy
val_in_distribution_expert/top5_accuracy
val_in_distribution_expert/samples

val_latest/loss
val_latest/top1_accuracy
val_latest/top3_accuracy
val_latest/top5_accuracy
val_latest/samples
val_latest/seconds

val_latest_expert/loss
val_latest_expert/top1_accuracy
val_latest_expert/top3_accuracy
val_latest_expert/top5_accuracy
val_latest_expert/samples
```

Expert namespaces do not report independent evaluation seconds because they
reuse the parent traversal. All metrics retain successful optimizer step as
their WandB horizontal axis.

## Components

`training/expert_validation.py` owns ZIP manifest discovery, CSV validation,
per-date score ranking, and stable expert episode-key sets. Keeping this logic
outside the training loop makes it independently testable.

`training/feature_cache.py` accepts the per-date expert-key mapping when
building splits and returns the two aligned expert masks.

`training/train.py` loads expert definitions before model construction,
passes them into split construction, and accumulates full/expert metrics in
each parent validation traversal.

`cfg/train.yaml` and `README.md` document the two new settings and metric
namespaces.

## Testing

Tests cover:

1. two-player score reconstruction from `min_score` and `sum_score`;
2. ranking across both player scores rather than episode maxima;
3. independent cutoffs for different dates;
4. inclusion of every episode tied at a cutoff;
5. rejection of missing, duplicate, non-finite, or non-two-player rows;
6. expert masks aligned with parent validation indices;
7. expert subsets contained completely within their parent sets;
8. one model forward per parent validation batch while both metric
   accumulators update;
9. correct CE and Top-1/3/5 totals for full and expert samples;
10. separate WandB namespaces with no expert evaluation-time metric.
