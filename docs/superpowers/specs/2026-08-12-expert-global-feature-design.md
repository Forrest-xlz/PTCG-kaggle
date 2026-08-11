# Expert-Conditioned Global Feature Design

## Goal

Add a binary expert-conditioning value to the model's global summary. Training
and validation labels use the existing per-date expert-validation ranking
logic, while Kaggle live inference exposes a manual boolean switch and normally
sets the condition to expert.

## Expert Definition

For each replay date independently, read every participant score from the
archive's `manifest.csv`. Use the same quantile, cutoff, and tie behavior as
`training.expert_validation.load_expert_date_info`, with
`train.expert_validation_ratio` as the ratio. A replay is expert when either
participant reaches that date's cutoff.

The label is replay-level because `manifest.csv` contains the lower score and
score sum but does not map the higher score to a player index. Consequently all
samples from an expert replay receive `is_expert = 1.0`, including a losing
player's samples when those samples are otherwise retained. All samples from
other replays receive `0.0`.

## Feature Layout

Append `is_expert` to the end of the existing global summary:

```text
global_summary[0:73] = existing features
global_summary[73]   = is_expert
```

Increase `GLOBAL_SUMMARY_DIM` from 73 to 74 everywhere. The existing global
summary projection MLP consumes all 74 values; no separate embedding or model
branch is added. All encoder, decoder, Transformer, loss, and validation logic
otherwise remains unchanged.

## Cache Construction and Configuration

Add `train_config: cfg/train.yaml` to the cache mapping. At cache startup, load
the referenced training configuration and obtain its `replay_episodes` path and
`expert_validation_ratio`. Build expert episode-key sets for every input source
date before worker processes start. Pass only the matching date's immutable key
set into each source worker, and append the label while preparing each feature
record.

The extracted JSONL format remains unchanged because it already contains the
episode ID. Extraction does not need to be repeated.

Record the expert ratio and new global layout in the cache feature signature.
Training and standalone validation build their expected signature using the
current `train.expert_validation_ratio`. A cache created with a different ratio
or the old 73-dimensional layout fails before training begins with a signature
or schema error. Increment the packed-cache schema version.

Changing `expert_validation_ratio` therefore requires rebuilding cache. It does
not require extraction.

## Training and Validation

Training reads the 74-dimensional cached global summary without runtime label
mutation. Validation splits continue using the existing expert episode sets and
are unchanged. Their expert flags use the same ratio and daily ranking as their
expert subgroup masks. Non-expert validation samples retain flag zero.

Old checkpoints are intentionally unsupported because the first global-summary
projection changes from 73 to 74 input columns. This experiment starts training
from a new model.

## Kaggle Inference

Add one editable parameter to the submission notebook:

```python
IS_EXPERT = True
```

The notebook's live feature builder appends `float(IS_EXPERT)` to every global
summary. `True` requests the learned expert-conditioned policy. `False` enables
an otherwise identical non-expert ablation. Include this parameter in notebook
validation and generated submission source; it is not inferred from live game
state.

## Error Handling and Tests

Fail clearly for a missing referenced training config, missing replay archive,
source date without expert keys, invalid expert ratio, cache/train ratio
mismatch, incorrect 74-dimensional summaries, and old cache schema.

Tests cover daily expert labeling, ties, expert/non-expert records, global
summary ordering and size, cache config inheritance, signature mismatch,
parallel source routing, model input shape, standalone validation compatibility,
and notebook `IS_EXPERT` behavior. Compile changed Python files and validate the
notebook JSON after implementation.
