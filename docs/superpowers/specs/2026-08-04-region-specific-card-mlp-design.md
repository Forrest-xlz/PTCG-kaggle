# Region-Specific Static Card MLP Design

## Goal

Allow the 54-dimensional static card features to use either the existing globally shared projection MLP or separate projection MLPs for different game regions.

Cards within one region share a projection. Different bench slots share the same regional projection, and a Pokemon, its tools, and its attached Energy cards also share the projection of that Active or Bench region.

## Configuration

Add the model setting:

```yaml
model:
  card_mlp_scope: region  # shared or region
```

- `shared` preserves the current single `card_feature_projection` behavior.
- `region` creates one static-card projection per region.
- `card_mlp_layers: 0` continues to disable every static-card contribution, independent of scope.

`ModelConfig` and training configuration validation reject values other than `shared` and `region`.

Old checkpoints do not contain `card_mlp_scope`. Checkpoint restoration treats a missing value as `shared`, retaining the old architecture and state-dict parameter names. New regional checkpoints record `card_mlp_scope: region` and restore the regional architecture automatically.

## Regions

The regional model uses these semantic regions:

1. own Bench;
2. opponent Bench;
3. own Active;
4. opponent Active;
5. own discard;
6. opponent discard;
7. own hand;
8. opponent hand;
9. own deck;
10. opponent deck;
11. own Prize cards;
12. opponent Prize cards;
13. Stadium;
14. cards currently being looked at;
15. unknown.

The encoder currently uses the relevant subset: both Bench regions, both Active regions, both discard regions, own hand, own deck, and Stadium. Decoder-only or currently rare regions still receive their own projections so they do not silently share semantics through the unknown projection.

## Encoder Mapping

The encoder vocabulary already assigns distinct ranges to:

- the four field layouts: own Bench, opponent Bench, own Active, and opponent Active;
- Pokemon, tool, and attached-Energy ranges inside each field layout;
- own discard, opponent discard, own hand, known own deck, and Stadium.

Extend its vocabulary metadata from only `index_to_card_id` to both:

- `index_to_card_id`;
- `index_to_card_region`.

All three card ranges inside one field layout receive the same region ID. Therefore all eight own Bench tokens share one static projection, while opponent Bench uses a different projection.

The learned encoder embedding table remains unchanged. Only the added static feature component becomes region-specific.

## Decoder Mapping

Do not change cached feature shapes or schemas. Infer candidate and target regions from fields already stored in every cached option:

- option type;
- player relation one-hot;
- area one-hot;
- in-play-area one-hot.

Rules:

- `PLAY` candidate maps to own hand even if the raw option area is absent.
- `RETREAT` target maps to own Active.
- Other candidates use their option area and relative player.
- `ATTACH` and `EVOLVE` targets use their in-play area and own-player relation, matching the current target-card extraction behavior.
- Stadium and looking areas ignore player relation.
- A real card whose region cannot be inferred maps to unknown.
- Card sentinels contribute a zero static embedding regardless of inferred region.

Candidate and target static card features reuse exactly the same regional MLPs as encoder cards. Candidate and target do not receive independent static-card MLPs.

## Model Implementation

For `shared`, retain `card_feature_projection` and the current two-dimensional projected table with shape `(card_count + 1, d_model)`.

For `region`, create a `ModuleDict` named `card_feature_projections`, keyed by stable region names. Each MLP has the existing architecture controlled by `card_mlp_layers`: the first layer maps `54 -> d_model`, and every additional layer maps `d_model -> d_model` with the existing activation placement.

Regional projection produces a table with shape:

```text
(region_count, card_count + 1, d_model)
```

The final card row remains an all-zero sentinel for every region. Encoder embedding-bag lookup combines region ID and Card ID. Decoder lookup indexes the same regional table with the inferred candidate and target regions.

## Training and Inference Integration

Update:

- `imitation_learning/model/network.py`;
- `imitation_learning/training/train.py`;
- `imitation_learning/cfg/train.yaml`;
- `imitation_learning/kaggle_submission_imitation_agent.ipynb`.

The training checkpoint already serializes the model configuration; the new field is included automatically after it is added to `ModelConfig`. The Kaggle notebook must define the same region constants, projection modules, lookup rules, and missing-field default so it can load both old shared checkpoints and new regional checkpoints.

No extraction or feature-cache rebuild is required because decoder regions are reconstructed from existing numeric and categorical option features, and encoder regions are determined from the fixed encoder vocabulary layout.

## Verification

Tests will verify:

- configuration accepts only `shared` and `region`;
- missing checkpoint scope defaults to `shared`;
- cards in different regions receive different projections in regional mode;
- all own Bench slots and their Pokemon/tool/Energy ranges use the same region projection;
- own and opponent Bench use different projections;
- decoder `PLAY`, `RETREAT`, ordinary area, Stadium, looking, and unknown rules select the expected projection;
- candidate and target reuse encoder regional projections;
- card sentinels remain zero;
- `card_mlp_layers: 0` disables static features in either scope;
- shared mode retains the old projection name and behavior;
- the submission notebook contains configuration and architecture parity with training code.
