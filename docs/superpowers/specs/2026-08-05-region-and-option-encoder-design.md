# Region and Option Encoder Design

## Goal

Replace the post-aggregation region MLPs and final Action MLP with explicit
component-token encoders. Variable-size card collections and Pokemon contents
must be represented as masked token sequences, while the outer policy model
continues to expose the same 26 encoder tokens to the main Transformer and the
same candidate-action logits to training.

## Scope

This change covers the internal construction of:

- own and opponent Bench tokens;
- own and opponent Active tokens;
- own and opponent discard tokens;
- own hand and known-deck tokens; and
- each raw decoder option embedding.

Player summaries, the global summary, stadium handling, static card and attack
projections, the main 26-token encoder, action enumeration, action-option
membership, the cross-attention-only decoder, training splits, losses, and
validation metrics remain unchanged.

## Configuration

Remove these fields and their implementations:

- `bench_token_mlp_layers`
- `active_token_mlp_layers`
- `discard_token_mlp_layers`
- `hand_token_mlp_layers`
- `deck_token_mlp_layers`
- `action_mlp_layers`
- `region_token_mlp_residual`

Add these non-negative integer fields:

- `bench_region_encoder_layers`
- `active_region_encoder_layers`
- `discard_region_encoder_layers`
- `hand_region_encoder_layers`
- `deck_region_encoder_layers`
- `option_encoder_layers`

Every internal encoder inherits `d_model`, `num_heads`, `d_feedforward`, and
`norm_mode` from the main model. It uses zero dropout. Own and opponent modules
use the same configured depth but independent weights, matching the existing
side-specific region modules. Transformer layers always retain their standard
residual connections; there is no configurable residual switch.

## Encoder Component Cache

The feature cache stores explicit ragged component sequences instead of one
pre-aggregated sparse bag per outer token. Each cached sample contains flattened
component arrays plus an offset array with a final end offset:

- component kind: Pokemon card, current HP, appearance state, Tool card,
  Energy card, or ordinary area card;
- component ID: Card ID or appearance-state ID where applicable;
- component scalar: `hp / 400` for HP and `1` for categorical/card components;
- component offsets: boundaries for the outer encoder-token component lists.

Duplicate physical cards remain duplicate component tokens. Card tokens use
full-scale embeddings; the old `0.25` area-card and `0.5` Energy-card bag
weights are not carried into the token representation. HP is the only scaled
component in this design.

The mmap representation stays ragged. The collator/model pads only the selected
batch and builds boolean padding masks, so disk and resident CPU memory are not
proportional to the maximum sequence length. The cache schema and signature are
bumped. Existing extracted winner-only JSONL is sufficient, but all feature
caches must be rebuilt.

## Bench and Active Sequences

Each present Pokemon produces:

```text
[Pokemon Card, current HP, appearThisTurn, Tool cards..., Energy cards...]
```

No CLS token is used.

- Depth zero: sum all present components and return the sum.
- Positive depth: pad and mask the sequence, run the appropriate internal
  encoder, and return position zero (the Pokemon Card output).

The Pokemon Card, Tool, and Energy roles use distinguishable learned Card ID
embeddings and the appropriate static-card projection. Current HP uses a
learned feature vector scaled by `hp / 400`. Appearance uses the existing
three-state embedding. Max HP, effective energy types, pre-evolution cards,
special conditions, and other temporary effects are not added by this change.

Missing Bench or Active positions have empty component sequences and remain
masked in the main 26-token encoder.

## Card-Collection Sequences

Discard, hand, and known-deck regions contain one token per physical card:

```text
[Card 1, Card 2, ..., Card N]
```

- Depth zero: do not create CLS; sum the Card tokens. An empty collection
  returns a zero vector.
- Positive depth: prepend a learned CLS owned by that region module, apply the
  masked internal encoder, and return the CLS output. An empty collection is a
  one-token CLS sequence.

Own and opponent discard use independent encoders and CLS parameters. Own hand
and own deck each have their own encoder and CLS. These component sequences are
treated as unordered multisets and receive no positional embeddings.

## Raw Option Sequences

Each raw engine option is built from these possible component tokens:

```text
option type
select context
candidate Card ID plus candidate static-card projection
target Card ID plus target static-card projection
Attack ID plus static-attack projection
projected 76-dimensional numeric features
```

Missing candidate, target, and Attack components are omitted/masked. Type,
context, and numeric components are always present. Static projection MLPs and
the option-numeric projection MLP remain because they map source features into
`d_model`; only the post-sum Action MLP is removed.

- Depth zero: do not create CLS; sum all present option components.
- Positive depth: prepend one shared learned option CLS, run the shared masked
  option encoder, and return its CLS output.

Candidate actions containing multiple raw options continue to sum their raw
option embeddings. There is no second action encoder and no Action MLP.

## Outer Model Data Flow

Internal region outputs fill the existing outer layout:

```text
0..7 own Bench, 8..15 opponent Bench,
16 own Active, 17 opponent Active,
18 own summary, 19 opponent summary,
20 own discard, 21 opponent discard,
22 own hand, 23 own deck,
24 stadium, 25 global summary
```

The main Transformer encoder processes these 26 tokens as before. Each summed
candidate action independently cross-attends to that main encoder output; there
is still no self-attention between candidate actions.

## Training and Kaggle Inference

Training YAML, checkpoint serialization, model construction, and the embedded
Kaggle notebook implementation use the new field names and identical data flow.
The notebook constructs ragged region components directly from a live
observation, pads only the current inference batch, and loads all internal
encoder and CLS parameters from the checkpoint.

Old checkpoints are architecture-incompatible and are not silently upgraded.
Replay extraction does not need to be rerun. Feature-cache construction must be
rerun because the explicit component layout changes the cache schema.

## Verification

Behavioral tests cover:

- depth-zero sum behavior without creating or using CLS;
- positive-depth CLS behavior for card collections and raw options;
- positive-depth Pokemon-position output for Bench and Active;
- masking invariance when padded components change;
- independent own/opponent parameters;
- missing and empty regions;
- preservation of duplicate card instances;
- YAML validation and checkpoint configuration round trips;
- cache-schema rejection of stale shards; and
- training/notebook architecture parity.

Runtime verification should rebuild a smoke cache, run a small forward/backward
batch at depth zero and depth one, and validate the Kaggle notebook top to
bottom when a Python environment with the engine is available.
