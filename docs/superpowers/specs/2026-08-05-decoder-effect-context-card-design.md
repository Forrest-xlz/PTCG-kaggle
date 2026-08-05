# Decoder Effect and Context Card Design

## Goal

Add `SelectData.effect` and `SelectData.contextCard` to every decoder option so
the policy can identify both the source of the current effect and the card the
current selection concerns. Each role must have an independent learned Card ID
embedding and a corresponding projection of the existing 54-dimensional static
card features.

## Feature layout

Increase `OPTION_CATEGORICAL_DIM` from 5 to 7 and use this fixed order:

1. option type
2. select context
3. candidate card ID
4. target card ID
5. attack ID
6. effect card ID
7. context card ID

The effect and context-card IDs are read once from `obs.select` and repeated for
all options in that selection. A missing, negative, or out-of-range Card ID uses
the existing `card_count` sentinel, matching candidate and target handling.

The 76-dimensional numeric option feature layout remains unchanged.

## Model encoding

Add independent learned embeddings:

- `option_effect_embedding`
- `option_context_card_embedding`

Both use `card_count + 1` rows and use `card_count` as the padding row. Their
outputs are added to the existing option representation.

The static-card contribution follows `card_mlp_scope`:

- `shared`: effect and context card use the existing shared 54-to-`d_model`
  projection.
- `region`: extend the static projection roles with `effect` and
  `context_card`, giving each an independent 54-to-`d_model` projection.
- `card_mlp_layers: 0`: both static contributions are zero, while their learned
  Card ID embeddings remain active.

Candidate and target region resolution remains unchanged. Effect and context
card do not require an `AreaType` because they are decision-level semantic
roles, not per-option source or target locations.

The final option input is the sum of the existing components plus:

```text
effect Card ID embedding
+ contextCard Card ID embedding
+ effect static-card projection
+ contextCard static-card projection
```

## Cache compatibility

Increase the packed feature-cache schema version and update the decoder layout
signature. Existing packed caches must be rebuilt because their categorical
rows contain five values rather than seven.

Replay extraction does not need to be repeated: extracted JSONL records retain
the complete observation, including `select.effect` and `select.contextCard`.

Existing checkpoints are not strictly compatible because the model gains two
embedding tables and, in region scope, two static projection modules. The
baseline is expected to be retrained after rebuilding the feature cache.

## Validation

Add focused tests that verify:

- feature extraction writes effect and context-card IDs in the documented
  categorical columns;
- missing cards use the sentinel;
- the network adds independent ID and static contributions for both roles;
- shared and region-specific static projection modes both produce valid option
  tensors;
- packed-cache validation accepts seven categorical values per option and
  rejects the obsolete schema.

Run the focused tests first, then the available model/cache test suite. Also
perform a small decoder forward pass to verify output shapes and finite logits.
