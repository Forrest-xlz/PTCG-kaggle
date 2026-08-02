# Effect and Context-Card Option Embeddings

## Goal

Expose `select.effect` and `select.contextCard` to the policy decoder so it can distinguish which card caused an effect from which card is the focus of the current sub-selection.

## Scope

This change only extends option-query construction. It does not add encoder tokens, self-attention between actions, new losses, or changes to action enumeration.

## Feature representation

Append two Card IDs to every option's categorical feature row:

1. `effect_card_id`: `obs.select.effect.id`, or `card_count` when absent or invalid.
2. `context_card_id`: `obs.select.contextCard.id`, or `card_count` when absent or invalid.

The categorical layout changes from five to seven fields:

```text
[option_type, select_context, candidate_card_id, target_card_id,
 attack_id, effect_card_id, context_card_id]
```

## Model representation

Add two independent learned embeddings:

```text
option_effect_embedding:       Embedding(card_count + 1, d_model)
option_context_card_embedding: Embedding(card_count + 1, d_model)
```

Both use `padding_idx=card_count`, matching candidate and target. Missing values therefore contribute a zero learned vector.

The shared 54-dimensional static card projection is also added for both roles. The resulting option query is:

```text
existing option query
+ learned effect-card embedding
+ learned context-card embedding
+ shared static effect-card projection
+ shared static context-card projection
```

The learned role embeddings remain independent so the same Card ID can mean different things as effect source, current context, candidate, or target. The static projection remains globally shared across all card roles.

## Cache and inference compatibility

Increase `OPTION_CATEGORICAL_DIM` from 5 to 7 and increment the packed feature-cache schema. Existing feature caches and checkpoints are incompatible and must not be reused. Replay extraction does not need to be repeated.

Apply the same feature layout and model construction to the Kaggle submission notebook so exported checkpoints load with strict state-dict validation.

## Validation

Tests must cover:

- effect and context cards both present with different Card IDs;
- only one of the two fields present;
- both fields absent and mapped to the zero-padding row;
- categorical cache shape changing from five to seven;
- Kaggle notebook containing both embeddings and the seven-field layout.

Static checks must confirm training, cache generation, model code, and submission inference use the same categorical dimension and field order.
