# Option Location and Auxiliary Feature Redesign

## Goal

Make every spatial option use the same coordinate system as the encoder while
removing numeric fields whose integer values have no ordinal meaning. Preserve
the existing action enumeration, policy-only BC objective, encoder/decoder
architecture, shared static card projection, and static attack projection.

This design supersedes the 16-field numeric contract in
`2026-08-01-option-embedding-design.md`.

## Shared encoder-token locations

The existing 20 encoder tokens receive one trainable shared location embedding
each. The location embedding is added directly to the encoder token before the
Transformer encoder.

The locations retain the current fixed token order:

1. own bench slots 0 through 4;
2. opponent bench slots 0 through 4;
3. own active and opponent active;
4. own-player and opponent-player summaries;
5. own discard and opponent discard;
6. own hand;
7. known own deck;
8. stadium;
9. global summary.

Option feature extraction resolves up to two location IDs: the candidate/source
location and the target location. The option embedding directly adds both rows
from the same shared location table:

```text
option += location[candidate_location] + location[target_location]
```

There are no candidate/target location projection layers. Candidate and target
card IDs continue to use their existing separate embedding tables, so their
semantic roles remain separately represented even though location addition is
commutative.

`area`, `index`, `inPlayArea`, and `inPlayIndex` are used during feature
extraction to resolve location IDs. `TOOL_CARD`, `ENERGY_CARD`, and `ENERGY`
options resolve to the host Pokemon token while the candidate Card ID identifies
the selected attachment. `ATTACK` and `RETREAT` resolve to the current player's
active token.

Hidden opponent zones without a dedicated token use the opponent-player summary
location. Equivalent own-player zones use the own-player summary. Options with
no spatial entity, such as YES, NO, NUMBER, and END, add no location vector; the
model does not learn a separate no-location embedding.

## Revised option auxiliary features

The existing five categorical identities remain unchanged:

1. option type;
2. selection context;
3. candidate Card ID;
4. target Card ID;
5. attack ID.

The old 16 numeric fields are replaced by the following 28-dimensional
auxiliary vector:

| Feature | Width | Representation |
| --- | ---: | --- |
| `number` | 1 | normalized scalar, unchanged `/ 6` scale |
| `count` | 1 | normalized scalar, unchanged `/ 10` scale |
| source `area` | 13 | one-hot: not applicable plus AreaType 1-12 |
| special condition | 6 | one-hot: not applicable plus five conditions |
| attack damage | 1 | normalized scalar |
| super-effective state | 3 | one-hot: not applicable, false, true |
| resisted state | 3 | one-hot: not applicable, false, true |

`number` and `count` remain scalars because they are quantities with meaningful
ordering. They do not receive separate presence flags because option type tells
the model whether each field is applicable.

The following old fields are removed from model input:

- raw `index`;
- own-player flag;
- `toolIndex`;
- `energyIndex`;
- `inPlayArea`;
- `inPlayIndex`;
- option-list relative position;
- combined linked-entity flag;
- normalized candidate card type.

Raw indices remain available only inside feature extraction where needed to
resolve Card IDs and encoder locations. Card type remains available through the
globally shared 54-dimensional static card features.

The matchup calculation is corrected to use the attacking Pokemon's energy type
instead of its broad CardType. Super-effective and resisted states are marked
not applicable for non-attack options.

## Attack features

Normalized attack damage remains in the auxiliary option vector as requested.
No additional attack-energy fields are added to that vector. The existing
static attack projection is otherwise unchanged, including its current attack
energy-cost information, because the rest of the option embedding logic remains
in scope unchanged.

## Storage and model flow

The cache stores compact categorical IDs for area, special-condition state,
matchup states, and the two encoder locations. One-hot expansion happens during
the model forward pass rather than being stored as float vectors for every
sample. This preserves one-hot semantics without multiplying cache size for the
multi-million-sample dataset.

The expanded 28 values pass through one `Linear(28, d_model)` layer and are
added to the existing option embedding. Learned embeddings and shared static
card/attack projections remain additive.

The feature-cache schema and feature signature are incremented. Existing replay
extraction remains valid; only feature caches must be rebuilt. The Kaggle
submission notebook must use the same feature contract, shared location table,
and location resolution.

## Validation

Targeted checks cover:

- all 20 encoder tokens receive the correct shared location row;
- hand, discard, active, bench, attached-card, attack, and retreat options
  resolve to the intended candidate/target locations;
- hidden zones fall back to the correct player-summary location;
- non-spatial options add no location vector;
- missing and poison special-condition states are distinct;
- non-attack, false, and true matchup states are distinct;
- removed numeric fields no longer affect option features;
- training and submission feature signatures remain identical.
