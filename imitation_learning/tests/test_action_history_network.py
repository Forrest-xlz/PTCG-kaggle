from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.attack_features import ATTACK_FEATURE_DIM  # noqa: E402
from model.card_features import CARD_FEATURE_DIM  # noqa: E402
from model.network import (  # noqa: E402
    ATTACK_DYNAMIC_DIM,
    HISTORY_STEPS,
    HISTORY_STRUCTURAL_DIM,
    OPTION_NUMERIC_DIM,
    POKEMON_DYNAMIC_DIM,
    ModelConfig,
    PTCGTransformer,
)


def _model(mode: str) -> PTCGTransformer:
    config = ModelConfig(
        card_count=8,
        attack_count=5,
        encoder_size=22_000,
        d_model=8,
        num_heads=2,
        d_feedforward=16,
        history_encoding=mode,
    )
    return PTCGTransformer(
        config,
        torch.zeros((config.card_count, CARD_FEATURE_DIM)),
        torch.zeros((config.attack_count, ATTACK_FEATURE_DIM)),
    )


@pytest.mark.parametrize(
    ("mode", "tokens"),
    [("off", 28), ("basic", 29), ("structural", 29), ("full", 29)],
)
def test_history_mode_controls_encoder_token_count(mode: str, tokens: int) -> None:
    assert _model(mode).encoder_token_count == tokens


def test_history_embeddings_do_not_share_decoder_parameters() -> None:
    model = _model("full")
    assert (
        model.history_option_type_embedding.weight.data_ptr()
        != model.option_type_embedding.weight.data_ptr()
    )
    assert "numeric" not in inspect.signature(model.encode_history).parameters


def test_enabled_history_requires_sequence_projection() -> None:
    with pytest.raises(ValueError, match="when history_encoding is enabled"):
        ModelConfig(
            card_count=8,
            attack_count=5,
            history_encoding="basic",
            history_sequence_mlp_layers=0,
        )
    ModelConfig(
        card_count=8,
        attack_count=5,
        history_encoding="off",
        history_sequence_mlp_layers=0,
    )


def test_history_encoder_returns_one_token_per_sample() -> None:
    model = _model("full")
    batch_size = 2
    rows = 3
    token = model.encode_history(
        select_type=torch.zeros((batch_size, HISTORY_STEPS), dtype=torch.long),
        select_context=torch.zeros(
            (batch_size, HISTORY_STEPS), dtype=torch.long
        ),
        valid=torch.tensor([[0, 1, 1], [0, 0, 0]], dtype=torch.long),
        categorical=torch.zeros((rows, 11), dtype=torch.long),
        structural=torch.zeros((rows, HISTORY_STRUCTURAL_DIM), dtype=torch.long),
        pokemon_dynamic=torch.zeros((rows, POKEMON_DYNAMIC_DIM)),
        attack_dynamic=torch.zeros((rows, ATTACK_DYNAMIC_DIM)),
        option_offsets=torch.tensor([0, 0, 2, 3, 3, 3, 3]),
    )
    assert token.shape == (batch_size, model.config.d_model)
    torch.testing.assert_close(token[1], torch.zeros_like(token[1]))


def test_current_decoder_still_has_five_numeric_inputs() -> None:
    assert OPTION_NUMERIC_DIM == 5
