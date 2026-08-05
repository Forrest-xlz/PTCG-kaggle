# Routed Option Features and Option-Token MLP Design

## Goal

Replace the decoder's mixed 76-dimensional option feature vector with
semantically separated categorical embeddings, routed Pokemon dynamics, and
attack dynamics. Preserve information required to distinguish legal options,
remove duplicated or positional shortcuts, and make post-sum option processing
configurable.

## Decoder Inputs

Each raw engine option is converted into three groups:

1. categorical IDs that are embedded directly;
2. two routed Pokemon dynamic slots that are projected together;
3. attack-only dynamic features that are projected separately.

The old generic `numeric` matrix and `option_numeric_mlp_layers` are removed.

### Categorical IDs

The categorical option tensor contains:

- `option_type`;
- `select_context`;
- `candidate_card_id`;
- `target_card_id`;
- `attack_id`;
- `number`;
- `count`;
- `player_relation`;
- `area`;
- `in_play_area`;
- `special_condition_type`.

Every field has a dedicated embedding. Missing values use a reserved padding
index whose embedding is zero. `number` and `count` retain their integer
semantics as categorical values rather than continuous scalars. Both use the
same explicit encoding: index 0 is missing, engine values 0 through 60 map to
indices 1 through 61, and values above 60 map to overflow index 62. Each table
therefore has 63 rows. Negative non-missing values are rejected as malformed
input.

`player_relation` has three values: missing, own, and opponent. This remains an
option-level embedding because non-Pokemon card selections can still refer to
either player.

`index`, `toolIndex`, `energyIndex`, and `inPlayIndex` remain available to the
feature extractor for resolving concrete cards and Pokemon, but they are not
fed directly to the model. The relative position of an option is also removed.

The following former numeric fields are removed because their information is
represented elsewhere:

- `has_entity`, represented by padding IDs and dynamic-presence masks;
- candidate card type, present in the 54-dimensional static card table;
- duplicate attack damage, present in the static attack table;
- raw and one-hot encodings of area, in-play area, special condition, player
  relation, candidate type, matchup, and entity presence.

## Routed Pokemon Dynamics

The extractor constructs two fixed semantic slots:

- `primary`: the Pokemon directly performing or receiving the operation;
- `secondary`: a second Pokemon only when the action represents a relation
  between two Pokemon.

If an action involves exactly one Pokemon, it always occupies `primary`.
`secondary` is never populated by itself.

### Routing

| Option type | Primary | Secondary |
| --- | --- | --- |
| `ATTACK` | own Active attacker | opponent Active default defender |
| `RETREAT` | own Active | missing |
| `ABILITY` | referenced in-play Pokemon, if any | missing |
| `ATTACH` | Pokemon at `inPlayArea/inPlayIndex` | missing |
| `EVOLVE` | Pokemon at `inPlayArea/inPlayIndex` | missing |
| `TOOL_CARD` | host Pokemon at `area/index` | missing |
| `ENERGY_CARD`, `ENERGY` | host Pokemon at `area/index` | missing |
| `CARD`, `DISCARD` | referenced card if it is an in-play Pokemon | missing |
| all other types | missing | missing |

For `TOOL_CARD`, `ENERGY_CARD`, and `ENERGY`, the dynamic router resolves the
host Pokemon independently. It must not reuse `candidate_card_id`, which points
to the attached Tool or Energy card after `toolIndex` or `energyIndex` has been
resolved.

### Per-Pokemon Features

Each slot has 23 values:

1. presence;
2. current HP divided by 400;
3. current maximum HP divided by 400;
4. lost HP divided by 400;
5. current HP divided by current maximum HP;
6. Tool count divided by 4;
7. attached Energy-card count divided by 10;
8. effective Energy count divided by 10;
9. twelve effective-Energy-type counts, each divided by 10;
10. appeared this turn;
11. is Active;
12. is owned by the observing player.

Missing slots are 23 zeros. The two slots are concatenated in the fixed order
`[primary, secondary]`, producing 46 values, then projected with one
`Linear(46, d_model)` layer. The projection output is multiplied by
`primary_present OR secondary_present`; therefore the result is exactly zero
when neither Pokemon exists, even though the linear layer has a bias.

## Attack Dynamics

Attack dynamics are populated only for an `ATTACK` option with a valid attack,
own Active Pokemon, and opponent Active Pokemon. The six values are:

1. attack-dynamic presence;
2. clipped printed base-damage-to-target-current-HP ratio;
3. whether printed base damage is at least the target's current HP;
4. target HP remaining after printed base damage, divided by 400;
5. whether the own Active Pokemon's type matches the opponent Active Pokemon's
   printed weakness;
6. whether it matches the opponent Active Pokemon's printed resistance.

The ratio is `min(base_damage / max(target_hp, 1), 4) / 4`. Remaining HP is
`max(target_hp - base_damage, 0) / 400`.

These are explicitly printed-base-damage features. They do not claim to model
variable attack text, temporary modifiers, prevention effects, bench targets,
or exact resolved damage.

`energy_deficit` is not included. The engine only exposes attack options after
checking that their effective energy cost is satisfied, so it would be nearly
constant zero. Effective attached energies are already present in the routed
Pokemon features, and printed energy requirements remain in the static attack
feature table.

The six values are projected with one `Linear(6, d_model)` layer. Its output is
multiplied by the attack-presence mask, producing exactly zero for non-attack
options.

## Static Card and Attack Features

Candidate and target Card IDs keep their learned embeddings and their existing
54-dimensional static-card projections. An absent candidate or target uses the
zero padding row.

Attack IDs keep their learned embedding and the existing 14-dimensional static
attack projection:

- printed damage divided by 300;
- twelve required-energy-type counts;
- total printed energy requirement divided by 5.

The new six-dimensional attack dynamic projection is additional and represents
the interaction between the selected attack and the current Active matchup.

## Option Token Construction

For each individual engine option, all available branches are summed:

```text
option type embedding
+ select context embedding
+ number embedding
+ count embedding
+ player relation embedding
+ area embedding
+ in-play area embedding
+ special-condition embedding
+ candidate Card-ID embedding
+ candidate static-card projection
+ target Card-ID embedding
+ target static-card projection
+ attack-ID embedding
+ static-attack projection
+ routed-Pokemon dynamic projection
+ attack dynamic projection
```

The result optionally passes through a post-sum option-token MLP controlled by:

```yaml
model:
  option_token_mlp_layers: 0
```

Layer semantics match the project's existing projection MLP convention:

- `0`: identity; use the embedding sum directly;
- `1`: `Linear(d_model, d_model)`;
- `2`: `Linear -> ReLU -> Linear`;
- additional layers insert another `ReLU -> Linear` pair.

No residual connection or LayerNorm is added by this setting.

The MLP is applied independently to each option before composite actions are
formed. Existing composite-action behavior remains unchanged: all option tokens
belonging to an action are summed with `embedding_bag`, and the resulting action
token is passed to decoder cross-attention. There is no action-to-action
self-attention.

## Configuration and Compatibility

Remove `model.option_numeric_mlp_layers` from configuration, validation,
checkpoint metadata, documentation, training construction, and submission
inference code. Add `model.option_token_mlp_layers`, validated as an integer
greater than or equal to zero.

The model configuration stored in checkpoints remains the source of truth for
inference architecture. Submission code must construct all new embeddings and
projections from that stored configuration.

The cached option representation changes shape and meaning. Increment the
feature-cache schema version and reject old caches with the existing clear
schema error. Training data extraction does not need to be repeated, but the
feature cache must be rebuilt.

## Verification

Verification should cover:

- every routing row above, including host-Pokemon resolution for attached cards;
- missing primary and secondary slots;
- exact-zero masked projections when Pokemon or attacks are absent;
- distinct encodings for `NUMBER` values and Energy `count` values;
- removal of direct positional indices from model inputs;
- option-token MLP behavior at depths zero, one, and two;
- unchanged composite-action summation;
- cache-schema rejection for previous caches;
- checkpoint round-trip and submission-side architecture reconstruction.
