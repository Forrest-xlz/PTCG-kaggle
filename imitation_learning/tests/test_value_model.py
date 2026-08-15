from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.attack_features import ATTACK_FEATURE_DIM
from model.card_features import CARD_FEATURE_DIM
from model.network import ModelConfig, PTCGTransformer
from value.model import PTCGEncoderBackbone, PTCGValueModel


def _policy() -> PTCGTransformer:
    config = ModelConfig(
        card_count=8,
        attack_count=4,
        encoder_size=22_000,
        d_model=8,
        num_heads=2,
        d_feedforward=16,
        encoder_layers=1,
        decoder_layers=1,
        history_encoding="off",
    )
    return PTCGTransformer(
        config,
        torch.zeros((config.card_count, CARD_FEATURE_DIM)),
        torch.zeros((config.attack_count, ATTACK_FEATURE_DIM)),
    )


def _encoder_inputs(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    token_count = 26
    words = batch_size * token_count
    return (
        torch.zeros(words, dtype=torch.long),
        torch.ones(words),
        torch.arange(words, dtype=torch.long),
        torch.zeros((batch_size, 18), dtype=torch.long),
        torch.zeros((batch_size, 69)),
        torch.zeros((batch_size, 71)),
        torch.zeros((batch_size, 73)),
        torch.zeros((batch_size, 3), dtype=torch.long),
        torch.zeros((batch_size, 3), dtype=torch.long),
        torch.zeros((batch_size, 3), dtype=torch.long),
        torch.zeros((0, 11), dtype=torch.long),
        torch.zeros((0, 8), dtype=torch.long),
        torch.zeros((0, 46)),
        torch.zeros((0, 6)),
        torch.zeros(batch_size * 3 + 1, dtype=torch.long),
    )


def test_encoder_backbone_loads_every_policy_encoder_weight() -> None:
    policy = _policy()
    backbone = PTCGEncoderBackbone.from_policy_state(
        policy.config,
        policy.card_feature_table,
        policy.attack_feature_table,
        policy.state_dict(),
    )

    assert not any(
        name.startswith(("decoder", "option_", "action_input_norm"))
        for name, _ in backbone.named_parameters()
    )
    for name, tensor in backbone.state_dict().items():
        torch.testing.assert_close(tensor, policy.state_dict()[name])


def test_encoder_backbone_rejects_missing_policy_weight() -> None:
    policy = _policy()
    state = dict(policy.state_dict())
    state.pop("encoder.layers.0.self_attn.in_proj_weight")

    with pytest.raises(ValueError, match="missing encoder tensors"):
        PTCGEncoderBackbone.from_policy_state(
            policy.config,
            policy.card_feature_table,
            policy.attack_feature_table,
            state,
        )


def test_value_model_uses_global_token_and_starts_at_zero() -> None:
    policy = _policy()
    backbone = PTCGEncoderBackbone.from_policy_state(
        policy.config,
        policy.card_feature_table,
        policy.attack_feature_table,
        policy.state_dict(),
    )
    model = PTCGValueModel(backbone, head_layers=2, dropout=0.0)

    encoded = backbone.encode_state(*_encoder_inputs())
    values = model(*_encoder_inputs())

    assert encoded.shape == (26, 2, 8)
    assert values.shape == (2,)
    torch.testing.assert_close(values, torch.zeros(2))


def test_value_model_tanh_bounds_after_nonzero_initialization() -> None:
    policy = _policy()
    backbone = PTCGEncoderBackbone.from_policy_state(
        policy.config,
        policy.card_feature_table,
        policy.attack_feature_table,
        policy.state_dict(),
    )
    model = PTCGValueModel(backbone, head_layers=1, dropout=0.0)
    torch.nn.init.constant_(model.output.weight, 100.0)
    torch.nn.init.constant_(model.output.bias, 100.0)

    values = model(*_encoder_inputs())

    assert torch.all(values <= 1.0)
    assert torch.all(values >= -1.0)


@pytest.mark.parametrize("layers", [0, -1])
def test_value_head_requires_at_least_one_layer(layers: int) -> None:
    policy = _policy()
    backbone = PTCGEncoderBackbone.from_policy_state(
        policy.config,
        policy.card_feature_table,
        policy.attack_feature_table,
        policy.state_dict(),
    )
    with pytest.raises(ValueError, match="head_layers"):
        PTCGValueModel(backbone, head_layers=layers, dropout=0.0)
