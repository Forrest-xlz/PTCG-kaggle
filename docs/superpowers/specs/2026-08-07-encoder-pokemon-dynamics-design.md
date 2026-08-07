# Encoder Pokemon Dynamics Design

## Goal

Improve encoder state observability by binding runtime Pokemon state directly
to each Active/Bench token, while retaining player-level summaries and keeping
the feature additions independently ablatable.

## Scope

This change adds:

- a shared 38-dimensional runtime Pokemon projection for all 18 field tokens;
- one independent immediate-pre-evolution Card ID embedding;
- 25 player-summary features covering bench capacity, effective Energy, HP,
  and whole-field aggregates;
- packed-cache, training, checkpoint, and Kaggle-inference parity.

It does not change the policy decoder, action-history representation,
Transformer layers, validation splits, or loss.

## Configuration

The model configuration gains two booleans:

```yaml
pokemon_dynamic_embedding: true
pre_evolution_embedding: true
```

Both default to `false` for old-checkpoint compatibility. The packed cache
always stores the superset, so either switch can be changed without another
cache rebuild after the schema upgrade.

## Per-Pokemon Runtime Features

One shared base constructor produces the decoder-compatible 23 values. The
encoder extends that base with special-condition and two-attack readiness
fields, producing 38 values per Pokemon token; the decoder remains 23-dimensional.

### Existing decoder-compatible fields: 23

```text
0      present
1      current_hp / 400
2      max_hp / 400
3      lost_hp / 400
4      current_hp / max_hp
5      tool_count / 4
6      energy_card_count / 10
7      effective_energy_count / 10
8-19   each of 12 effective Energy counts / 10
20     appear_this_turn
21     is_active
22     is_own
```

`energyCards` counts physical attached Energy cards. `energies` contains the
effect-resolved Energy units and types. Both are retained.

### Active special conditions: 5

```text
23     poisoned
24     burned
25     asleep
26     paralyzed
27     confused
```

These values come from `PlayerState`. They are populated only for Active
Pokemon; every Bench Pokemon receives five zeros.

### Readiness of the first two attacks: 10

For each of the first two Attack IDs listed by the card, store:

```text
attack_present
base_damage / 300
required_energy_count / 5
missing_energy_count / 5
is_energy_ready
```

A missing attack contributes five zeros.

Energy matching handles non-Colorless requirements first. An exact Energy
type satisfies itself, Rainbow satisfies any colored requirement, and Team
Rocket satisfies Psychic or Darkness. Remaining Energy units of any type then
satisfy Colorless requirements. `missing_energy_count` is the number of
unmatched requirements, and `is_energy_ready` is one exactly when that count
is zero.

## Encoder Integration

When enabled, one globally shared projection maps runtime features:

```text
pokemon_dynamic_encoder_projection: Linear(38, d_model)
```

For each of the 18 field tokens:

```text
pokemon_token += pokemon_dynamic_encoder_projection(runtime_features)
```

Absent slots are masked to an exact zero contribution. The projection is
shared across own/opponent and Active/Bench tokens; token position continues
to express role and ownership.

## Immediate Pre-Evolution Embedding

For each field Pokemon, cache the Card ID of the most recent pre-evolution:

```python
previous_card_id = (
    pokemon.preEvolution[-1].id
    if pokemon is not None and pokemon.preEvolution
    else card_count
)
```

The model owns a separate table:

```text
Embedding(card_count + 1, d_model, padding_idx=card_count)
```

This table does not share weights with the main Pokemon, Tool, Energy, decoder,
or static-card embeddings. Its padding row contributes exactly zero.

## Player Summary Extension

The existing common 54 values retain their exact order and normalization.
Twenty-five new values are appended.

### Capacity and Active effective Energy: 3

```text
bench_max / 8
bench_count / max(bench_max, 1)
active_effective_energy_count / 10
```

The ratio is zero if `bench_max` is zero.

### Per-Bench additions: 16

For each of eight Bench slots:

```text
bench_i_max_hp / 400
bench_i_effective_energy_count / 10
```

Missing slots contribute zeros.

### Aggregate additions: 6

```text
bench_effective_energy_count_sum / 80
bench_min_current_hp / 400
all_in_play_energy_card_count_sum / 90
all_in_play_effective_energy_count_sum / 90
all_in_play_current_hp_sum / 3600
all_in_play_max_hp_sum / 3600
```

`bench_min_current_hp` is zero when there is no Bench Pokemon. Existing
physical-Energy, Bench-current-HP, and Bench-max-HP aggregates remain present.

The new dimensions are:

```text
common player summary: 79
own summary:            79 + 15 = 94
opponent summary:       79 + 17 = 96
```

## Packed Cache

The cache schema is incremented. Every sample stores:

- runtime Pokemon features with shape `[18, 38]`;
- immediate-pre-evolution IDs with shape `[18]`;
- the expanded own, opponent, and global summary arrays.

These arrays are stored regardless of the two model switches. Existing
extracted winner-only JSONL contains all required observations, so raw replay
extraction is not repeated. Existing packed feature caches must be rebuilt.

## Training and Checkpoints

Training transfers runtime Pokemon values as floating point and pre-evolution
IDs as `long`. `ModelConfig` stores both switches, making checkpoints
architecture-self-describing. Old checkpoints without the fields load with
both additions disabled.

## Kaggle Inference

The submission notebook mirrors the feature construction, configuration,
modules, and forward arguments. It derives all runtime features from the
current observation and reconstructs the two optional modules exclusively
from checkpoint configuration.

## Verification

Focused checks cover:

- exact 38-column order and absent-Pokemon zeros;
- exact, Rainbow, Team Rocket, and Colorless Energy matching;
- zero-padded missing attacks and missing Bench slots;
- 79/94/96 summary dimensions and empty-Bench minimum HP;
- immediate `preEvolution[-1]` selection and zero sentinel;
- cache round trips and collation;
- all four combinations of the two model switches;
- old checkpoint defaults;
- project/notebook feature parity and notebook syntax.
