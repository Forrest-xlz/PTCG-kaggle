# Deck Validation, Replay Scaling, and Submission Design

## Scope

This change adds three deck-focused validation subsets, deterministic
replay-level training-data scaling, and a checkpoint-configured Kaggle
submission notebook. Existing base and expert validation sets remain intact.

## Cache Representation

Feature cache schema version increases from 3 to 4. Every cached sample stores
a `uint64 deck_key` computed from the canonical sorted list of its complete
60-card deck. The source training JSONL already stores the acting winner's
deck, so replay extraction does not need any new fields.

The canonical key uses a stable 64-bit cryptographic digest over all 60 card
IDs with multiplicity preserved. Card order is irrelevant, while different
card counts or IDs produce a different canonical input. This adds eight bytes
per cached sample and avoids rescanning compressed JSONL at training startup.

Existing schema-3 feature caches are incompatible and must be rebuilt.
Replay JSONL extraction can be reused, although the user intends to rerun both
extraction and cache generation for newly added data.

## Configured Decks and Validation Subsets

`cfg/train.yaml` adds `train.top_decks`, a nested list containing one or more
complete decks:

```yaml
train:
  top_decks:
    - [7, 7, 7, 7, ...]
```

Every configured deck must:

- be a list of exactly 60 integers;
- contain only non-negative card IDs;
- be canonicalized by sorting before its key is calculated.

Multiple configured decks are combined with OR semantics: a sample belongs to
the top-deck cohort when its `deck_key` matches any configured deck.

Three validation subsets are added:

1. `val_in_distribution_top_deck`: in-distribution validation samples whose
   acting winner used a configured deck.
2. `val_latest_top_deck`: latest-date validation samples whose acting winner
   used a configured deck.
3. `val_latest_expert_top_deck`: intersection of
   `val_latest_top_deck` and the existing latest-date expert episode subset.

Each subset logs CE loss, Top-1, Top-3, Top-5 accuracy, and sample count.
Empty configured-deck subsets are treated as configuration/data errors and
fail at startup.

The in-distribution dataset is still traversed once per evaluation event, and
the latest-date dataset is still traversed once. Each model forward produces
logits shared by the base set and every aligned subgroup mask, so none of the
new subsets adds a model forward pass.

W&B namespaces are:

- `val_in_distribution_top_deck/*`
- `val_latest_top_deck/*`
- `val_latest_expert_top_deck/*`

Existing `val_in_distribution_expert/*` and `val_latest_expert/*` namespaces
remain unchanged.

## Replay-Level Training Data Scaling

`cfg/train.yaml` adds:

```yaml
train:
  train_replay_ratio: 0.1
  train_replay_seed: 42
```

The latest date is first held out entirely. The configured
`validation_ratio` is then applied by replay to older dates. Only after both
validation sets are fixed is `train_replay_ratio` applied to the remaining
training replays.

Sampling uses a stable episode hash mixed with `train_replay_seed`. Every
eligible replay is selected with the configured probability, making the
result approximate by replay count rather than forcing an exact sample count.
All samples from a selected replay are retained. The selected training subset
is fixed for the entire run and is reshuffled by sample within each epoch as
before.

`train_replay_ratio` must be in `(0, 1]`; `1.0` preserves all eligible training
replays. Startup output and W&B data metrics record:

- eligible training sample and replay counts before scaling;
- selected training sample and replay counts;
- realized replay and sample ratios.

This makes data-scale experiments reproducible without leaking validation
replays or splitting decisions from the same replay.

## Kaggle Submission Notebook

`kaggle_submission_imitation_agent.ipynb` remains a standalone packaging
notebook. Its parameter cell exposes only:

- the submission deck's 60 card IDs;
- the exact uploaded checkpoint path;
- the exact uploaded `cg` directory path.

It no longer silently selects the lexicographically first checkpoint or
`cg` folder under `/kaggle/input`.

The notebook does not duplicate training architecture settings. It reads the
checkpoint's saved `config`, constructs `ModelConfig` and `PTCGTransformer`
from it, then loads `checkpoint["model"]` strictly. This automatically adapts
to:

- `d_model`;
- feed-forward dimension;
- attention head count;
- encoder layer count;
- decoder layer count;
- PreNorm or PostNorm.

Before packaging, the notebook validates the checkpoint structure and prints
the resolved architecture. Missing config fields, missing model weights, or
state-dict shape mismatches fail immediately.

The generated archive contains:

- `main.py`;
- `model.pt`;
- `deck.csv`;
- `cg/`.

No project source tree is required as an uploaded Kaggle Dataset.

## Error Handling

Training fails early when:

- a cache uses an older schema;
- any configured deck is malformed;
- a required replay manifest or expert mapping is unavailable;
- any requested deck validation subset is empty;
- replay sampling produces an empty training set;
- ratios are outside their valid ranges.

The notebook fails early when configured paths do not exist, the deck is not
exactly 60 cards, the checkpoint does not contain `model` and `config`, or the
saved weights cannot be loaded into the checkpoint-defined architecture.

## Verification

Focused tests cover:

- stable deck keys with order independence and multiplicity sensitivity;
- schema-4 cache round trips;
- all three deck-subset masks and their intersections;
- replay-level training scaling without splitting a replay;
- multi-subgroup validation metrics from one set of model forwards;
- YAML fields and validation;
- notebook JSON validity, explicit path parameters, and checkpoint-driven
  model construction.

The notebook should be executed top-to-bottom in Kaggle after attaching a real
checkpoint and `cg` Dataset, because the local workspace does not provide the
Kaggle filesystem or runtime.
