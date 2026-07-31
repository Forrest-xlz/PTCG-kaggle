# Numeric Summary Tokens Design

## Goal

Replace the existing sparse player-summary and global-summary embeddings with
three dense, notebook-compatible numeric projections:

- own-player summary: `Linear(60, d_model)`
- opponent-player summary: `Linear(62, d_model)`
- global/select summary: `Linear(73, d_model)`

The reference notebook's 125 numeric features remain semantically intact,
except that both prize counts, `select.type`, and `select.context` become
one-hot vectors. Own and opponent discard piles become two independent card
tokens instead of being mixed into the player-summary tokens.

## Encoder Token Layout

The encoder layout is fixed internally at 20 tokens. It is not exposed as a
YAML parameter.

| Token | Meaning | Representation |
|---:|---|---|
| 0-4 | Own bench slots 0-4 | Existing Pokémon card token |
| 5-9 | Opponent bench slots 0-4 | Existing Pokémon card token |
| 10 | Own active Pokémon | Existing Pokémon card token |
| 11 | Opponent active Pokémon | Existing Pokémon card token |
| 12 | Own-player summary | `Linear(60, d_model)` |
| 13 | Opponent-player summary | `Linear(62, d_model)` |
| 14 | Own discard pile | Card `EmbeddingBag` |
| 15 | Opponent discard pile | Card `EmbeddingBag` |
| 16 | Own hand | Existing card `EmbeddingBag` |
| 17 | Known own deck | Existing card `EmbeddingBag` |
| 18 | Stadium | Existing card `EmbeddingBag` |
| 19 | Global and current-selection summary | `Linear(73, d_model)` |

The two discard tokens preserve card multiplicity and use weight `0.25` per
card, matching the previous discard and hand weighting. Each discard zone has
its own learned Card ID embedding block. All card tokens continue to add the
globally shared 54-dimensional static-card projection.

The previous three sparse embeddings for own summary, opponent summary, and
global summary are removed rather than added to the new projections.

## Original 125-Dimensional Contract

The original order is:

```text
0-38    own player
39-77   opponent player
78-85   global battle state
86-92   current selection
93-107  own deck/prize tracking
108-124 opponent revealed-card statistics
```

### Per-Player Features

Player-local indices `p0-p38` map to original indices `0-38` for the own
player and `39-77` for the opponent.

| Local index | Own index | Opponent index | Feature |
|---:|---:|---:|---|
| p0 | 0 | 39 | deck count / 60 |
| p1 | 1 | 40 | hand count / 20 |
| p2 | 2 | 41 | discard count / 60 |
| p3 | 3 | 42 | remaining prize count |
| p4 | 4 | 43 | bench count / 5 |
| p5 | 5 | 44 | active Pokémon exists |
| p6 | 6 | 45 | poisoned |
| p7 | 7 | 46 | burned |
| p8 | 8 | 47 | asleep |
| p9 | 9 | 48 | paralyzed |
| p10 | 10 | 49 | confused |
| p11 | 11 | 50 | active current HP / 400 |
| p12 | 12 | 51 | active maximum HP / 400 |
| p13 | 13 | 52 | active attached-energy count / 10 |
| p14 | 14 | 53 | active tool count / 4 |
| p15 | 15 | 54 | active pre-evolution count / 2 |
| p16 | 16 | 55 | active retreat cost / 5 |
| p17 | 17 | 56 | active maximum attack damage / 300 |
| p18 | 18 | 57 | active attack count / 4 |
| p19 | 19 | 58 | active weakness enum index / 12 |
| p20 | 20 | 59 | active resistance enum index / 12 |
| p21 | 21 | 60 | bench slot 0 exists |
| p22 | 22 | 61 | bench slot 0 current HP / 400 |
| p23 | 23 | 62 | bench slot 0 energy count / 5 |
| p24 | 24 | 63 | bench slot 1 exists |
| p25 | 25 | 64 | bench slot 1 current HP / 400 |
| p26 | 26 | 65 | bench slot 1 energy count / 5 |
| p27 | 27 | 66 | bench slot 2 exists |
| p28 | 28 | 67 | bench slot 2 current HP / 400 |
| p29 | 29 | 68 | bench slot 2 energy count / 5 |
| p30 | 30 | 69 | bench slot 3 exists |
| p31 | 31 | 70 | bench slot 3 current HP / 400 |
| p32 | 32 | 71 | bench slot 3 energy count / 5 |
| p33 | 33 | 72 | bench slot 4 exists |
| p34 | 34 | 73 | bench slot 4 current HP / 400 |
| p35 | 35 | 74 | bench slot 4 energy count / 5 |
| p36 | 36 | 75 | total bench energy / 20 |
| p37 | 37 | 76 | total bench current HP / 2000 |
| p38 | 38 | 77 | total bench maximum HP / 2000 |

The prize-count scalar `p3` is replaced by a seven-dimensional one-hot vector
for counts `0-6`. Therefore each 39-dimensional player block becomes 45
dimensions before player-specific extensions.

### Own Deck/Prize Tracking

Original indices `93-107` are appended only to the own-player summary.

| Original index | Feature |
|---:|---|
| 93 | remaining Pokémon count / 4 |
| 94 | remaining Item count / 4 |
| 95 | remaining Tool count / 4 |
| 96 | remaining Supporter count / 4 |
| 97 | remaining Stadium count / 4 |
| 98 | remaining Basic Energy count / 4 |
| 99 | remaining Special Energy count / 4 |
| 100 | remaining Basic Pokémon count / 4 |
| 101 | remaining Stage 1 count / 4 |
| 102 | remaining Stage 2 count / 4 |
| 103 | any remaining Pokémon ex |
| 104 | any remaining Mega Pokémon ex |
| 105 | estimated remaining-card count / 60 |
| 106 | Pokémon share among remaining cards |
| 107 | Energy share among remaining cards |

The own-player projection input is therefore:

```text
39 - 1 + 7 + 15 = 60 dimensions
```

### Opponent Revealed-Card Statistics

Original indices `108-124` are appended only to the opponent-player summary.

| Original index | Feature |
|---:|---|
| 108 | revealed Pokémon count / 10 |
| 109 | revealed Item count / 10 |
| 110 | revealed Tool count / 10 |
| 111 | revealed Supporter count / 10 |
| 112 | revealed Stadium count / 10 |
| 113 | revealed Basic Energy count / 10 |
| 114 | revealed Special Energy count / 10 |
| 115 | any revealed Pokémon ex |
| 116 | any revealed Mega Pokémon ex |
| 117 | any revealed Tera Pokémon |
| 118 | revealed Pokémon count / 5 |
| 119 | total revealed-card count / 20 |
| 120 | average revealed-Pokémon HP / 400 |
| 121 | average revealed-Pokémon stage / 2 |
| 122 | opponent discard size / 20 |
| 123 | opponent bench size / 5 |
| 124 | opponent in-play energy count / 10 |

Indices 41 and 122 are both retained because they come from different parts of
the reference contract and use different normalization constants.

The opponent-player projection input is:

```text
39 - 1 + 7 + 17 = 62 dimensions
```

### Global Battle State

Original indices `78-85` remain in the global token.

| Original index | Feature |
|---:|---|
| 78 | turn / 100 |
| 79 | turn action count / 100 |
| 80 | first player relative to the agent: unknown=-1, opponent=0, own=1 |
| 81 | supporter played this turn |
| 82 | stadium played this turn |
| 83 | energy attached this turn |
| 84 | retreated this turn |
| 85 | `yourIndex` |

### Current Selection

Original indices `86-92` remain in the global token.

| Original index | Feature | New encoding |
|---:|---|---|
| 86 | `select.type` | 11-way one-hot for enum values 0-10 |
| 87 | `select.context` | 49-way one-hot for enum values 0-48 |
| 88 | `select.minCount` / 6 | scalar |
| 89 | `select.maxCount` / 6 | scalar |
| 90 | `select.remainDamageCounter` / 20 | scalar |
| 91 | `select.remainEnergyCost` / 10 | scalar |
| 92 | legal option count / 64 | scalar |

The global projection input is:

```text
8 + 11 + 49 + 5 = 73 dimensions
```

## Model Components

The three projections are independent single linear layers with bias and no
activation:

```text
own_summary_projection: Linear(60, d_model)
opponent_summary_projection: Linear(62, d_model)
global_projection: Linear(73, d_model)
```

They do not share weights because their dimensions and semantics differ.
Cross-token interaction remains the responsibility of the Transformer
encoder.

## Data Flow and Cache Impact

Feature extraction builds the three dense numeric vectors and the 17 sparse
card tokens. Training batches provide both groups to the model. The model
projects the numeric vectors, evaluates the card-aware `EmbeddingBag` tokens,
places all outputs in the fixed 20-token order, and sends the result through
the existing Transformer encoder.

Replay extraction data remains sufficient and does not need to be regenerated.
The feature-cache schema and encoder layout do change, so feature caches must
be rebuilt. The cache signature must reject older 24-token caches.

The Kaggle submission notebook must build the same numeric vectors, use the
same 20-token order, and restore all three projection layers from the
checkpoint.

## Validation

Tests must verify:

- exact 60, 62, and 73 input dimensions;
- every original index `0-124` is assigned exactly once before categorical
  expansion;
- prize-count, selection-type, and selection-context one-hot ranges;
- five bench tokens per player and exactly 20 encoder tokens;
- independent own/opponent discard tokens with multiplicity and weight `0.25`;
- old player/global sparse embeddings are absent;
- old feature caches are rejected by signature;
- training and Kaggle inference produce identical features and token order.
