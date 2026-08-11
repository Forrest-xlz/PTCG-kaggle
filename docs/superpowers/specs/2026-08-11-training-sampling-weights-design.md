# Training Sampling Weights Design

## Goal

Add deterministic per-epoch base sampling plus extra sampling for winning
samples from expert replays, configured exact decks, and their intersections.
Preserve existing replay-level splits, loser augmentation eligibility, and
global epoch shuffle.

## Configuration

Add the following nested mapping under `train`:

```yaml
sampling:
  base_sample_ratio: 0.8       # Random fraction of the full training set used each epoch.
  expert_ratio: 0.05            # Daily top-player fraction used for expert replay cutoff.
  expert_extra_weight: 1.0      # Extra copies of expert-replay winning samples.
  deck_extra_weights:           # deckN follows train.top_decks order; omitted decks add nothing.
    deck1: 1.0
    deck2: 0.2
    deck4: 1.5
  expert_deck_extra_weights:    # Extra expert winners using deckN.
    deck1: 2.0
    deck2: 1.0
```

`base_sample_ratio` must be in `(0, 1]`. All extra weights must be finite
numbers greater than or equal to `0.0`. Deck keys
must have the exact form `deckN`, where `N` is between 1 and the number of
configured `train.top_decks`. Omitted deck keys mean no extra samples.
`expert_ratio` must be finite and in `(0, 1]`.

## Expert and Deck Membership

Read each replay archive's existing `manifest.csv`. Reconstruct the higher
participant score from `min_score` and `sum_score`, matching current expert
validation parsing. For each date independently, rank all participant scores
and calculate the top-`expert_ratio` cutoff with the same quantile and tie
behavior as expert validation. An episode is expert when either participant's
score reaches that date's cutoff. This remains an episode-level label because
the cache and manifest do not map the higher score back to an acting player.
The sampling ratio is independent of `expert_validation_ratio` so training
weights can be tuned without changing validation subsets.

Exact-deck membership uses each cached sample's acting-player `deck_key` and
the stable keys of `train.top_decks`. `deck1`, `deck2`, and later numbers refer
to that YAML list's order.

Construct masks only after isolation, latest-date, and in-distribution
validation replays have been fixed. Masks align with `splits.train`. Extra
sampling requires `player_result == WIN`. Losing samples, including loser
augmentation, can enter an epoch only through base sampling and are never
duplicated by extra rules.

## Base Sampling and Independent Extra Sampling

Start each epoch with a without-replacement random sample of
`floor(base_sample_ratio * train_sample_count)` indices from the complete final
training split. Independently apply three extra rule families to their complete
eligible subsets, not merely to members selected by the base sample:

1. Expert winning samples add `expert_extra_weight`.
2. Each exact deck's winning samples add its `deck_extra_weight`.
3. Each expert/exact-deck winning intersection adds
   its `expert_deck_extra_weight`.

Overlapping rules add rather than replace or maximize one another. For each
extra rule with weight `w`, append `floor(w)` complete copies of its eligible
indices, then append a without-replacement random sample of size
`floor((w - floor(w)) * eligible_count)`. A weight of `2.0` adds two complete
copies, while `0.2` adds 20 percent of eligible indices. More explicitly, `0`
adds nothing, `1.0` adds one full extra copy, and `1.5` adds one full copy plus
a random 50 percent.

Use a deterministic RNG derived from `train.seed + epoch_index`. Fractional
subsets therefore change between epochs but reproduce under identical config,
including epoch-level resume. Concatenate all base and added parts, globally
shuffle the combined index array once, and feed it to existing batches without
a second shuffle.

## Epoch Length, Scheduling, and Limits

The combined count is constant across epochs because base and fractional sample
sizes use `floor`, although selected members change. Compute
`samples_per_epoch`, `steps_per_epoch`, cosine schedule length, and warmup
validation from this count. If `train.max_samples` is set, apply it after the
combined global shuffle; scheduler length uses the same truncated count.

At startup print the base sampling ratio and selected count, each date's expert
ratio cutoff,
eligible winning-sample count and added count for every active
expert/deck/intersection rule, complete training count, combined epoch count,
and effective multiplier. Record equivalent data metrics when experiment
tracking is enabled. Existing training, validation, and epoch metrics retain
their names.

## Data and Compatibility

No replay extraction or feature-cache rebuild is required. The feature cache
already contains episode keys, acting-player deck keys, dates, and player
results. Add a bounded-memory metadata lookup aligned to `splits.train`; do not
materialize observations or decoder features while constructing masks.

Invalid ratios, weights, deck names, empty active rule subsets, or
metadata misalignment fail before model training begins. An omitted or
zero-weight deck rule is inactive and is not an error.

## Tests

Cover YAML validation and comments, daily ratio-based episode selection,
winner-only aligned masks, arbitrary valid `deckN` keys, rejection beyond the
configured deck list, base ratio counts, additive overlaps, exact integer and
fractional extra counts, per-epoch deterministic variation, global permutation
integrity, loser base-only sampling, combined scheduler length, and resume
reproducibility.
