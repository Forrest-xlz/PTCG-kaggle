# Date-Weighted Training Sampling Design

## Goal

Add deterministic date-based sample weighting to behavior-cloning training so that newer training dates can contribute more samples per epoch than older dates, without changing any validation split or cache format.

## Scope

- Apply weighting only after isolation, latest-date, in-distribution, replay-ratio, and loser-augmentation selection have produced `splits.train`.
- Weight individual samples, not whole replays.
- Apply the same date weight to winning samples and eligible losing samples already present in `splits.train`.
- Build the weighted training index once at process startup. Every epoch globally shuffles the same weighted index.
- Do not modify extraction, feature caching, validation membership, model architecture, or evaluation metrics.

## Configuration

Add the following mapping under `train` in `cfg/train.yaml`:

```yaml
date_sampling:
  enabled: true
  mode: power          # linear or power
  seed: 42             # Fractional-copy sampling; reconstructed on resume.
  linear:
    start: 0.5         # Earliest training date weight.
    end: 1.2           # Latest training date weight.
  power:
    start: 0.5         # Earliest training date weight.
    end: 1.2           # Latest training date weight.
    exponent: 2.0      # Shape of interpolation over calendar time.
```

All weights must be finite and non-negative. `power.exponent` must be finite and strictly positive. Both curve mappings are validated, while only the mapping selected by `mode` affects sampling. When `enabled: false`, the effective training indices equal `splits.train` exactly.

## Date Coordinate and Curve

Only dates represented in the final `splits.train` participate in the curve. The latest-date validation date is already absent from `splits.train` and therefore cannot become the curve endpoint.

Convert each `month.day` tuple to an ordinal calendar date using a fixed non-leap reference year. Let the earliest training date have coordinate `x=0` and the latest training date have coordinate `x=1`; dates between them use their actual elapsed-day position rather than their rank in the list.

For a linear curve:

```text
w(x) = start + (end - start) * x
```

For a power curve:

```text
w(x) = start + (end - start) * x^exponent
```

If only one training date remains, use the configured `end` weight because that date is also the newest available training data.

## Sample Expansion

For each date weight `w`:

```text
full_copies = floor(w)
fraction = w - full_copies
```

Every sample from that date is inserted `full_copies` times. One additional copy is inserted independently with probability `fraction`, using a deterministic NumPy generator seeded by `date_sampling.seed`.

Examples:

- `0.5`: approximately half of that date's samples appear once.
- `1.0`: every sample appears once.
- `1.2`: every sample appears once, plus an independently sampled 20% extra copy.
- `2.5`: every sample appears twice, plus an independently sampled 50% third copy.

Sampling happens once during startup. Epoch shuffling uses the existing `train.seed + epoch_index` behavior. Resuming with the same data and configuration reconstructs exactly the same weighted index.

An enabled configuration that produces zero total training samples is rejected with a clear error.

## Dataset Date Lookup

`MmapFeatureDataset` already stores each shard's date and global sample boundaries. Add a vectorized method that maps arbitrary global sample indices to their shard dates/date ordinals. The weighted-index builder must not inspect or decode feature records.

## Training Integration

Immediately after `splits` are built:

1. Build `weighted_train_indices` from `splits.train` and date sampling settings.
2. Compute `samples_per_epoch`, `steps_per_epoch`, total scheduler steps, and warmup validation from the weighted index length.
3. Pass `weighted_train_indices` to the existing `dataset.iter_batches` loop.
4. Preserve `max_samples` semantics by applying it after weighting, as the iterator already does.

No per-batch curve or random sampling work is added, so steady-state training throughput is unchanged apart from intentionally processing more or fewer samples.

## Reporting

At startup print one line per training date containing:

```text
date_sampling_date=7.1 x=0.000000 weight=0.500000 source_samples=10000 weighted_samples=5012
```

Also print a summary containing mode, seed, original training samples, weighted training samples, and realized overall ratio. Log equivalent aggregate and per-date counts/weights to W&B under `data/date_sampling/*`.

Checkpoint model and optimizer state remain unchanged. Save the resolved YAML settings as currently done so the date sampling configuration is recorded with the run.

## Tests

Add focused tests covering:

- exact linear endpoint and midpoint weights using real calendar spacing;
- power interpolation and exponent behavior;
- a single training date using `end`;
- weights `0`, below `1`, above `1`, and above `2`;
- deterministic reconstruction from the same seed and differing fractional selection from another seed;
- winning and losing samples receiving identical treatment because weighting operates only on final indices;
- exclusion of the latest validation date by deriving endpoints only from final training indices;
- disabled mode preserving the original index array;
- settings validation and training integration using weighted length for scheduler steps.

## Compatibility and Migration

- No re-extraction is required.
- No feature-cache rebuild is required.
- Existing checkpoints remain loadable.
- Changing only date sampling settings changes the samples used per epoch but not model parameter compatibility.
