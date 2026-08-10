# Training Sampling Weights Design

## Goal

Add deterministic per-epoch oversampling for winning samples from expert
replays, configured exact decks, and their intersections. Preserve the full
base training set, existing replay-level splits, loser augmentation, and global
epoch shuffle.

## Configuration

Add the following nested mapping under `train`:

```yaml
sampling:
  expert_score_threshold: 1200  # Replay is expert when its higher player score reaches this value.
  expert_weight: 1.5            # Extra sampling for winning samples from expert replays.
  deck_weights:                 # deckN follows the order of train.top_decks; omitted decks use 1.0.
    deck1: 2.0
    deck2: 1.2
    deck4: 1.5
  expert_deck_weights:          # Extra sampling for winning samples in both expert and deckN.
    deck1: 3.0
    deck2: 2.0
```

All weights must be finite numbers greater than or equal to `1.0`. Deck keys
must have the exact form `deckN`, where `N` is between 1 and the number of
configured `train.top_decks`. Omitted deck keys mean weight `1.0`. The score
threshold must be finite.

## Expert and Deck Membership

Read each replay archive's existing `manifest.csv`. Reconstruct the higher
participant score from `min_score` and `sum_score`, matching current expert
validation parsing. An episode is expert when its higher score is at least
`expert_score_threshold`. This is an episode-level label because the cache and
manifest do not map the higher score back to an acting player.

Exact-deck membership uses each cached sample's acting-player `deck_key` and
the stable keys of `train.top_decks`. `deck1`, `deck2`, and later numbers refer
to that YAML list's order.

Construct masks only after isolation, latest-date, and in-distribution
validation replays have been fixed. Masks align with `splits.train`. Additional
sampling requires `player_result == WIN`. Existing losing samples, including
loser augmentation, remain in the base training indices once and are never
duplicated by these rules.

## Independent Additive Oversampling

Start each epoch with every base training index exactly once. Apply the three
rule families independently:

1. Expert winning samples add `expert_weight - 1`.
2. Each exact deck's winning samples add `deck_weight - 1`.
3. Each expert/exact-deck winning intersection adds
   `expert_deck_weight - 1`.

Overlapping rules add rather than replace or maximize one another. For each
rule with weight `w`, append `floor(w) - 1` complete copies of its eligible
indices, then append a without-replacement random sample of size
`floor((w - floor(w)) * eligible_count)`. A weight of `1.0` adds nothing; `2.0`
adds one complete copy; `1.2` adds 20 percent of eligible indices.

Use a deterministic RNG derived from `train.seed + epoch_index`. Fractional
subsets therefore change between epochs but reproduce under identical config,
including epoch-level resume. Concatenate all base and added parts, globally
shuffle the combined index array once, and feed it to existing batches without
a second shuffle.

## Epoch Length, Scheduling, and Limits

The oversampled count is constant across epochs because fractional sample sizes
use `floor`, although the selected members change. Compute `samples_per_epoch`,
`steps_per_epoch`, cosine schedule length, and warmup validation from this
weighted count. If `train.max_samples` is set, apply it after the combined
global shuffle; scheduler length uses the same truncated count.

At startup print the expert threshold, eligible winning-sample count and added
count for every active expert/deck/intersection rule, base sample count,
weighted epoch count, and effective multiplier. Record equivalent data metrics
when experiment tracking is enabled. Existing training, validation, and epoch
metrics retain their names.

## Data and Compatibility

No replay extraction or feature-cache rebuild is required. The feature cache
already contains episode keys, acting-player deck keys, dates, and player
results. Add a bounded-memory metadata lookup aligned to `splits.train`; do not
materialize observations or decoder features while constructing masks.

Invalid thresholds, weights, deck names, empty configured rule subsets, or
metadata misalignment fail before model training begins. An omitted deck rule
is not considered an error because it has effective weight `1.0`.

## Tests

Cover YAML validation and comments, threshold-based episode selection,
winner-only aligned masks, arbitrary valid `deckN` keys, rejection beyond the
configured deck list, additive overlaps, exact integer and fractional counts,
per-epoch deterministic variation, global permutation integrity, loser
non-duplication, weighted scheduler length, and resume reproducibility.
