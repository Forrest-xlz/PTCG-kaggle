from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from model.attack_features import ATTACK_FEATURE_DIM  # noqa: E402
from model.card_features import CARD_FEATURE_DIM  # noqa: E402
from model.network import (  # noqa: E402
    ENCODER_POKEMON_DYNAMIC_DIM,
    ModelConfig,
    PTCGTransformer,
)


def _model(**overrides) -> PTCGTransformer:
    config = ModelConfig(
        card_count=8,
        attack_count=4,
        d_model=8,
        num_heads=2,
        d_feedforward=16,
        **overrides,
    )
    return PTCGTransformer(
        config,
        torch.zeros((config.card_count, CARD_FEATURE_DIM)),
        torch.zeros((config.attack_count, ATTACK_FEATURE_DIM)),
    )


def test_encoder_pokemon_features_are_optional_and_shared() -> None:
    disabled = _model()
    assert disabled.encoder_pokemon_dynamic_projection is None
    assert disabled.pre_evolution_embedding is None

    enabled = _model(
        pokemon_dynamic_embedding=True,
        pre_evolution_embedding=True,
    )
    assert enabled.encoder_pokemon_dynamic_projection.in_features == 38
    assert enabled.encoder_pokemon_dynamic_projection.out_features == 8
    assert enabled.pre_evolution_embedding.weight.shape == (9, 8)
    assert ENCODER_POKEMON_DYNAMIC_DIM == 38


def test_absent_pokemon_masks_dynamic_projection_bias() -> None:
    model = _model(pokemon_dynamic_embedding=True)
    torch.nn.init.zeros_(model.encoder_pokemon_dynamic_projection.weight)
    torch.nn.init.constant_(model.encoder_pokemon_dynamic_projection.bias, 2)
    features = torch.zeros((1, 18, ENCODER_POKEMON_DYNAMIC_DIM))
    features[0, 1, 0] = 1

    projected = model.encoder_pokemon_dynamic_projection(features)
    projected = projected * features[:, :, :1].gt(0)
    assert torch.equal(projected[0, 0], torch.zeros(8))
    assert torch.all(projected[0, 1] == 2)

