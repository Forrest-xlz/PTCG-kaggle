from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.network import OPTION_NUMERIC_DIM, ModelConfig, PTCGTransformer
from training.precision import PrecisionContext


def tiny_model(
    norm_mode: str = "postnorm",
    **config_overrides,
) -> PTCGTransformer:
    return PTCGTransformer(
        ModelConfig(
            card_count=8,
            attack_count=4,
            encoder_size=160,
            d_model=8,
            num_heads=2,
            d_feedforward=16,
            encoder_layers=1,
            decoder_layers=1,
            norm_mode=norm_mode,
            **config_overrides,
        ),
        torch.zeros((8, 54), dtype=torch.float32),
        torch.zeros((4, 14), dtype=torch.float32),
    )


@pytest.mark.parametrize("norm_mode", ["prenorm", "postnorm"])
def test_normalization_modes_preserve_policy_shape(norm_mode: str) -> None:
    torch.manual_seed(3)
    model = tiny_model(norm_mode).eval()
    encoder_index = torch.tensor([1, 2], dtype=torch.int32)
    encoder_value = torch.tensor([1.0, 0.5])
    encoder_offset = torch.tensor([0, 1] + [2] * 24, dtype=torch.int32)
    pokemon_appear = torch.zeros((1, 18), dtype=torch.long)
    own_summary = torch.zeros((1, 69))
    opponent_summary = torch.zeros((1, 71))
    global_summary = torch.zeros((1, 73))
    option_categorical = torch.tensor(
        [[8, 0, 1, 8, 4], [13, 0, 8, 8, 1]], dtype=torch.long
    )
    option_numeric = torch.zeros((2, OPTION_NUMERIC_DIM))
    action_option_index = torch.tensor([0, 1], dtype=torch.long)
    action_option_offset = torch.tensor([0, 1, 2], dtype=torch.long)
    logits = model(
        encoder_index,
        encoder_value,
        encoder_offset,
        pokemon_appear,
        own_summary,
        opponent_summary,
        global_summary,
        option_categorical,
        option_numeric,
        action_option_index,
        action_option_offset,
    )
    assert logits.shape == (1, 2)
    assert model.encoder.layers[0].norm_first is (norm_mode == "prenorm")
    assert (model.encoder.norm is not None) is (norm_mode == "prenorm")


def test_combination_actions_sum_options_and_use_learned_empty_embedding() -> None:
    model = tiny_model()
    model.no_action_embedding.data.fill_(4)

    combined = model.combine_actions(
        torch.tensor([[1.0] * 8, [2.0] * 8]),
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 2, 3, 3]),
    )
    torch.testing.assert_close(
        combined,
        torch.tensor([[3.0] * 8, [1.0] * 8, [4.0] * 8]),
    )


def test_pokemon_appear_embedding_and_region_mlps_are_independent() -> None:
    model = tiny_model(
        pokemon_appear_embedding=True,
        bench_token_mlp_layers=1,
        active_token_mlp_layers=1,
        discard_token_mlp_layers=1,
        hand_token_mlp_layers=1,
        deck_token_mlp_layers=1,
    )

    assert model.pokemon_appear_embedding.weight.shape == (3, 8)
    assert torch.count_nonzero(model.pokemon_appear_embedding.weight[0]) == 0
    assert (
        model.own_bench_token_mlp.weight
        is not model.opponent_bench_token_mlp.weight
    )
    assert (
        model.own_active_token_mlp.weight
        is not model.opponent_active_token_mlp.weight
    )
    assert (
        model.own_discard_token_mlp.weight
        is not model.opponent_discard_token_mlp.weight
    )


def test_region_token_mlp_residual_switch() -> None:
    residual_model = tiny_model(
        hand_token_mlp_layers=1,
        region_token_mlp_residual=True,
    )
    direct_model = tiny_model(
        hand_token_mlp_layers=1,
        region_token_mlp_residual=False,
    )
    token = torch.ones((1, 1, 8))
    for model in (residual_model, direct_model):
        model.own_hand_token_mlp.weight.data.zero_()
        model.own_hand_token_mlp.bias.data.fill_(2)

    torch.testing.assert_close(
        residual_model._apply_token_mlp(
            token, residual_model.own_hand_token_mlp
        ),
        torch.full_like(token, 3),
    )
    torch.testing.assert_close(
        direct_model._apply_token_mlp(
            token, direct_model.own_hand_token_mlp
        ),
        torch.full_like(token, 2),
    )

def test_invalid_normalization_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="norm_mode"):
        tiny_model("sandwich")


def test_fp32_precision_is_supported_on_cpu() -> None:
    context = PrecisionContext("fp32", torch.device("cpu"))
    assert not context.scaler.is_enabled()
    assert context.autocast_enabled is False


@pytest.mark.parametrize("name", ["fp16", "bf16"])
def test_mixed_precision_is_rejected_on_cpu(name: str) -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        PrecisionContext(name, torch.device("cpu"))


def test_invalid_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="precision"):
        PrecisionContext("tf32", torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fp16_context_uses_grad_scaler() -> None:
    context = PrecisionContext("fp16", torch.device("cuda"))
    assert context.scaler.is_enabled()
    assert context.autocast_dtype == torch.float16


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
def test_bf16_context_does_not_use_grad_scaler() -> None:
    context = PrecisionContext("bf16", torch.device("cuda"))
    assert not context.scaler.is_enabled()
    assert context.autocast_dtype == torch.bfloat16
