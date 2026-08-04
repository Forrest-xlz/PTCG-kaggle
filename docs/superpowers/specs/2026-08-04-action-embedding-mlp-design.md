# Action Embedding MLP Design

## Goal

Add a configurable MLP to each final candidate action embedding after all
member option embeddings have been summed and before decoder cross-attention.

## Architecture

The decoder data flow becomes:

```text
option component embeddings
-> sum options belonging to each candidate action
-> add no_action_embedding for an empty action
-> shared Action MLP
-> cross-attention-only decoder layers
-> policy logit
```

All candidate actions share one Action MLP. It processes the complete action,
not each option separately. Candidate actions still do not self-attend to one
another.

## Configuration

Add one model setting:

```yaml
model:
  action_mlp_layers: 1
```

Reuse the existing `region_token_mlp_residual` boolean for both encoder region
MLPs and the Action MLP. Do not introduce another residual setting.

A depth of zero creates no Action MLP and leaves the combined action embedding
unchanged. A depth of one creates `Linear(d_model, d_model)`. Every additional
layer appends `ReLU` and `Linear(d_model, d_model)`.

When the MLP exists:

- residual enabled: `output = action + MLP(action)`;
- residual disabled: `output = MLP(action)`.

## Compatibility

`ModelConfig.action_mlp_layers` defaults to zero, so checkpoint dictionaries
created before this change construct no new parameters and remain loadable.
New training YAML sets the depth explicitly. Checkpoint serialization already
records model configuration and therefore needs no new mechanism.

This change does not alter extracted records, feature tensors, cache layout, or
cache signature. Neither replay extraction nor feature-cache construction needs
to be rerun.

The Kaggle submission notebook must reproduce the same module construction,
action-combination order, residual behavior, and zero-depth fallback.

## Validation

- depth zero is identity and creates no Action MLP parameters;
- positive depth creates the requested MLP structure;
- the MLP is applied after option summation and empty-action embedding;
- residual on/off implements the documented formulas;
- actions share MLP weights and remain independent decoder queries;
- old checkpoint dictionaries missing the field use depth zero;
- the Kaggle notebook restores the setting from checkpoint configuration.
