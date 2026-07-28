# Isolation Validation Integration Design

## Goal

Integrate the three reviewed deck-selection CSV files into training as
replay-level isolation validation sets:

- `val_deck_isolation`
- `val_archetype_isolation`
- `val_top_deck_archetype_isolation`

Isolation is applied before latest-date and in-distribution validation
splitting. The isolation union is absent from latest-date validation,
in-distribution validation, and training.

## Inputs

The `train` section of `cfg/train.yaml` owns the configuration:

```yaml
train:
  isolation_validation:
    deck_data: data/deck
    selections:
      deck_isolation: data/deck_isolation_selection.csv
      archetype_isolation: data/archetype_isolation_selection.csv
      top_deck_archetype_isolation: data/top_deck_archetype_isolation_selection.csv
```

`deck_data` contains the per-date `*.decks.csv` outputs produced by the deck
extractor. Each file contains both players' exact decks and the original
`date` and `episode_id`.

Every selection CSV contains a `card_ids` column. Each value must decode to
exactly 60 non-negative integer Card IDs. Duplicate exact decks within one
selection are collapsed.

The three selection files and the deck-data directory are required. This
baseline does not include an enable/disable flag.

## Isolation Replay Resolution

A new `training/isolation_validation.py` module resolves selection decks into
replays before cache splitting.

It:

1. Reads and validates each selection CSV.
2. Converts each selected 60-card multiset to the same stable exact-deck key
   used by the feature cache.
3. Scans the deck extractor CSV files once.
4. Marks a replay for a selection whenever either player's exact deck matches
   one of that selection's deck keys.
5. Returns per-selection, per-date sets of stable episode keys.

The resolver validates that every selected exact deck occurs in the deck
extract data and that every selection resolves to at least one replay.

The feature cache stores only the sample player's deck key, so matching only
the cache's `deck_key` would miss replays where the isolated deck is the
opponent. Resolving from the two-player deck CSV is therefore required.

No feature-cache schema change or cache rebuild is required.

## Split Precedence

Splitting operates on replay groups and follows this exact order:

1. Resolve the three isolation replay sets.
2. Build their replay union.
3. Remove the isolation union from all later split candidates.
4. From the remaining samples, hold out the numerically latest source date as
   `val_latest`.
5. From remaining older-date replays, select `validation_ratio` into
   `val_in_distribution` using the existing deterministic replay hash.
6. Apply `train_replay_ratio` only to the remaining replay groups.
7. Use the result as the training set.

The latest date remains the numerically latest cache source date. If removing
isolation replays makes its validation set empty, training startup fails.

The in-distribution validation ratio is measured against non-isolation,
non-latest replay candidates. This keeps it representative of the actual
training distribution.

## Overlap Semantics

The three isolation validation sets may overlap. A replay that satisfies two
isolation definitions belongs to both corresponding metrics.

Only the isolation union participates in exclusion. Therefore:

- isolation versus latest-date overlap is zero;
- isolation versus in-distribution overlap is zero;
- isolation versus training overlap is zero;
- latest-date, in-distribution, and training remain mutually exclusive;
- pairwise overlap among the three isolation sets is allowed and reported.

## Dataset Split Representation

`DatasetSplits` gains:

- one sorted global-index array for the isolation union;
- three boolean masks aligned with that union, keyed by the public validation
  namespace;
- per-split replay and sample counts needed for audit logging.

`MmapFeatureDataset.build_splits()` accepts the per-selection, per-date episode
key sets. It creates the union and aligned masks shard by shard before
latest-date and hash-based selection.

All isolation subsets, latest-date validation, in-distribution validation, and
training must contain at least one sample. Invalid or empty sets raise a
descriptive `ValueError`; there is no silent fallback.

Existing expert and top-deck subgroup masks remain scoped to
`val_in_distribution` and `val_latest`. They are not added to the three
isolation validations.

## Evaluation

At every configured evaluation step:

1. Run one forward pass over the isolation union.
2. Accumulate each sample into every aligned isolation mask it matches.
3. Log the three isolation results independently.
4. Evaluate `val_latest` with its existing subgroups.
5. Evaluate `val_in_distribution` with its existing subgroups.

The isolation union does not publish a combined model-quality metric. It is
only an inference optimization and a data-audit count.

Each isolation validation reports:

- cross-entropy loss;
- Top-1 accuracy;
- Top-3 accuracy;
- Top-5 accuracy;
- sample count.

## Wandb Namespaces

The three isolation validations are separate Wandb groups:

```text
val_deck_isolation/*
val_archetype_isolation/*
val_top_deck_archetype_isolation/*
```

They are registered independently with `optimizer_step` as their step metric.
No `val_isolation/*` metric namespace is created.

Data audit metrics under `data/*` include:

- each isolation set's replay and sample counts;
- isolation-union replay and sample counts;
- all three pairwise isolation replay overlaps;
- post-isolation latest-date, in-distribution, and training counts.

## Error Handling

Training startup fails before model construction when:

- a configured selection file or deck-data directory is missing;
- a selection CSV lacks `card_ids`;
- a `card_ids` value is malformed or is not a valid 60-card deck;
- a selected exact deck is absent from the deck extractor data;
- a configured isolation set maps to no cached samples;
- isolation removal empties latest-date validation;
- the later in-distribution validation or training set is empty;
- any forbidden split overlap is detected.

The error includes the affected selection namespace, file, or split.

## Verification

Unit coverage includes:

- selection CSV parsing and 60-card validation;
- either-player deck matching;
- selected-deck absence diagnostics;
- allowed overlap between isolation definitions;
- isolation-union exclusion from latest, in-distribution, and training;
- latest-date construction after isolation removal;
- in-distribution replay sampling from only the remaining older dates;
- one isolation-union evaluation producing three independent metric groups;
- YAML parsing for the nested isolation configuration;
- Wandb namespace registration without a combined isolation metric group.

## Scope Exclusions

This change does not:

- rebuild extraction output or feature caches;
- alter model architecture or checkpoint format;
- create expert subgroups inside isolation validation;
- make the three isolation validation sets mutually exclusive;
- change the three reviewed selection CSV files;
- add an option to disable isolation validation.
