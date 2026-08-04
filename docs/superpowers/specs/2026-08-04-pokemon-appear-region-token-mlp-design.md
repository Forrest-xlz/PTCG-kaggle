# Pokemon Appear-This-Turn and Region Token MLP Design

## Goal

Add the missing `Pokemon.appearThisTurn` state to every visible Active and
Bench Pokemon token. After the encoder embedding sum is constructed, optionally
process selected semantic regions with independent `d_model -> d_model` MLPs
before the Transformer encoder.

The change must preserve the existing 26-token encoder layout. It changes the
cached feature schema but does not change replay extraction.

## Appear-This-Turn Feature

The game API defines `Pokemon.appearThisTurn: bool` as true when that Pokemon
entered play during the current turn. It is not a flag for the first turn of
the game.

Feature extraction produces an integer tensor with 18 values per sample in
encoder token order:

1. own Bench slots 0 through 7;
2. opponent Bench slots 0 through 7;
3. own Active;
4. opponent Active.

Use three categorical values:

- `0`: no Pokemon exists in this token;
- `1`: a Pokemon exists and `appearThisTurn` is false;
- `2`: a Pokemon exists and `appearThisTurn` is true.

When `pokemon_appear_embedding` is true, the model owns one globally shared
`Embedding(3, d_model, padding_idx=0)` table. Values 1 and 2 are added to the
corresponding Pokemon token after the sparse embedding bag has produced its
`d_model` representation. Value 0 produces zero, so an empty Bench or Active
token receives no appear-state contribution. Missing Bench tokens remain
excluded by the existing encoder padding mask.

Using an explicit cached categorical tensor keeps the boolean semantics shared
across own/opponent and Bench/Active regions. It also avoids hiding this state
inside unrelated sparse-vocabulary offsets.

## Region Token MLP Configuration

Add the following model settings:

```yaml
model:
  pokemon_appear_embedding: true
  bench_token_mlp_layers: 0
  active_token_mlp_layers: 0
  discard_token_mlp_layers: 0
  hand_token_mlp_layers: 0
  deck_token_mlp_layers: 0
  region_token_mlp_residual: true
```

Each layer-count setting is an integer greater than or equal to zero. The
feature and residual settings are booleans. New training configurations enable
`pokemon_appear_embedding`; old checkpoint dictionaries that do not contain the
field resolve it to false so their parameter layout remains loadable.

Five settings control eight independently parameterized modules:

| Configuration | Independent modules |
| --- | --- |
| `bench_token_mlp_layers` | own Bench, opponent Bench |
| `active_token_mlp_layers` | own Active, opponent Active |
| `discard_token_mlp_layers` | own discard, opponent discard |
| `hand_token_mlp_layers` | own hand |
| `deck_token_mlp_layers` | own remaining-deck token |

The two sides use the same configured depth for a semantic region, but never
share MLP weights. Stadium and the three dense summary tokens do not use these
post-token MLPs.

## MLP Semantics

A depth of zero creates no module and leaves that region unchanged. A depth of
one creates one `Linear(d_model, d_model)`. Every additional layer appends
`ReLU` followed by `Linear(d_model, d_model)`.

For enabled modules:

- when `region_token_mlp_residual` is true, output is `token + MLP(token)`;
- when it is false, output is `MLP(token)`.

The residual switch is global and applies only to the eight configured region
modules. It does not alter the Transformer encoder or decoder residual paths.

## Encoder Data Flow

The encoder keeps this token order:

```text
0..7    own Bench
8..15   opponent Bench
16      own Active
17      opponent Active
18      own summary
19      opponent summary
20      own discard
21      opponent discard
22      own hand
23      own remaining deck
24      stadium
25      global summary
```

Processing occurs in this order:

1. build the existing sparse embedding-bag token representations, including
   learned Card ID embeddings and static-card projections;
2. add the shared appear-state embedding to tokens 0 through 17;
3. replace the three summary placeholders with their existing dense projected
   summaries;
4. apply each independent region MLP to its assigned token slice;
5. pass the resulting 26 tokens and the existing padding mask to the
   Transformer encoder.

Applying the MLP after all embedding contributions lets each region transform
the complete token rather than only the 54-dimensional static-card component.

## Cache and Training Integration

The packed cache gains an `encoder_pokemon_appear` integer field of shape
`(samples, 18)`. Store it with the smallest supported unsigned integer dtype.
Increment the cache schema version and update the feature signature so old
caches fail with an explicit incompatibility error instead of being
misinterpreted.

Replay extraction remains unchanged because the extracted observations already
contain `appearThisTurn`. Users must rerun `cache_features.py`, not
`training/extract.py`.

Training batches transfer the new tensor to the model together with the current
encoder tensors. Checkpoints record the feature switch, five depth settings,
and residual setting through the existing serialized `ModelConfig`.

Old checkpoint configuration dictionaries default
`pokemon_appear_embedding` to false, all five region depths to zero, and the
residual flag to true. Therefore the new modules are absent and the old state
dict remains loadable. They still require a cache matching the model invocation
path when used for training.

## Kaggle Submission Integration

The submission notebook must reproduce the same 18-value feature order, shared
appear-state embedding, region MLP construction, token indices, and residual
behavior. Architecture values continue to load entirely from the checkpoint.
The user only supplies the checkpoint, game engine, and deck configuration.

## Validation

Cover these behaviors:

- feature extraction emits absent, existing-false, and existing-true values in
  the documented token order;
- disabling `pokemon_appear_embedding` preserves the old checkpoint parameter
  layout;
- the shared appear embedding is added to all and only Pokemon tokens;
- empty Bench positions receive no appear embedding and remain masked;
- the five depth settings create eight separate parameter sets with the
  expected depths;
- own and opponent modules of the same region do not share parameters;
- zero depth is identity;
- residual on/off implements the documented formulas;
- cache schema rejects old packed caches;
- the Kaggle notebook contains the same feature and architecture logic;
- old checkpoint dictionaries missing the new settings disable the appear
  embedding and resolve to zero-depth region MLPs.
