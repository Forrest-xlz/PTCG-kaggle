from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.attack_features import ATTACK_FEATURE_DIM
from model.card_features import CARD_FEATURE_DIM
from model.features import OPTION_CATEGORICAL_DIM
from model.network import (
    ATTACK_DYNAMIC_DIM,
    GLOBAL_SUMMARY_DIM,
    HISTORY_STEPS,
    HISTORY_STRUCTURAL_DIM,
    OPPONENT_SUMMARY_DIM,
    OWN_SUMMARY_DIM,
    OPTION_NUMERIC_DIM,
    POKEMON_DYNAMIC_DIM,
    POKEMON_ENCODER_TOKENS,
    ModelConfig,
    PTCGTransformer,
)


def _config(**changes) -> ModelConfig:
    base = ModelConfig(
        card_count=10,
        attack_count=5,
        encoder_size=256,
        d_model=8,
        num_heads=2,
        d_feedforward=16,
        encoder_layers=1,
        decoder_layers=1,
        revealed_hand_token_mlp_layers=1,
    )
    return replace(base, **changes)


def _model(config: ModelConfig) -> PTCGTransformer:
    return PTCGTransformer(
        config,
        torch.zeros(config.card_count, CARD_FEATURE_DIM),
        torch.zeros(config.attack_count, ATTACK_FEATURE_DIM),
    )


class RevealedHandModelTests(unittest.TestCase):
    def test_revealed_hand_tokens_use_independent_mlp_weights(self) -> None:
        model = _model(_config())

        self.assertEqual(model.encoder_token_count, 28)
        self.assertIsNotNone(model.own_revealed_hand_token_mlp)
        self.assertIsNotNone(model.opponent_revealed_hand_token_mlp)
        self.assertIsNot(
            model.own_revealed_hand_token_mlp,
            model.opponent_revealed_hand_token_mlp,
        )

    def test_enabled_cls_adds_one_learned_unmasked_token(self) -> None:
        model = _model(_config(learnable_cls_token=True))
        own = torch.zeros(1, OWN_SUMMARY_DIM)
        opponent = torch.zeros(1, OPPONENT_SUMMARY_DIM)
        revealed = torch.tensor([[1, 0]])

        mask = model._encoder_padding_mask(
            own,
            opponent,
            revealed,
            history_valid=None,
        )

        self.assertEqual(model.encoder_token_count, 29)
        self.assertEqual(tuple(model.cls_token.shape), (1, 1, 8))
        self.assertEqual(mask.shape, (1, 29))
        self.assertEqual(mask[:, 26:28].tolist(), [[False, True]])
        self.assertEqual(mask[:, -1:].tolist(), [[False]])

    def test_disabled_cls_does_not_create_parameter(self) -> None:
        model = _model(_config(learnable_cls_token=False))

        self.assertEqual(model.encoder_token_count, 28)
        self.assertIsNone(model.cls_token)

    def test_new_configuration_fields_are_validated(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "revealed_hand_token_mlp_layers"
        ):
            _config(revealed_hand_token_mlp_layers=-1)
        with self.assertRaisesRegex(ValueError, "learnable_cls_token"):
            _config(learnable_cls_token=1)

    def test_forward_accepts_revealed_presence_and_cls(self) -> None:
        config = _config(learnable_cls_token=True)
        model = _model(config)
        revealed_base = 8 + 17 * config.card_count
        categorical = torch.tensor(
            [[0, 0, config.card_count, config.card_count,
              config.attack_count, 0, 0, 0, 0, 0, 0]],
            dtype=torch.long,
        )

        logits = model(
            torch.tensor([revealed_base + 7]),
            torch.tensor([1.0]),
            torch.tensor([0] * 27 + [1]),
            torch.zeros(1, POKEMON_ENCODER_TOKENS, dtype=torch.long),
            torch.zeros(1, OWN_SUMMARY_DIM),
            torch.zeros(1, OPPONENT_SUMMARY_DIM),
            torch.zeros(1, GLOBAL_SUMMARY_DIM),
            torch.tensor([[1, 0]]),
            torch.zeros(1, HISTORY_STEPS, dtype=torch.long),
            torch.zeros(1, HISTORY_STEPS, dtype=torch.long),
            torch.zeros(1, HISTORY_STEPS, dtype=torch.long),
            torch.empty(0, OPTION_CATEGORICAL_DIM, dtype=torch.long),
            torch.empty(0, HISTORY_STRUCTURAL_DIM, dtype=torch.long),
            torch.empty(0, POKEMON_DYNAMIC_DIM),
            torch.empty(0, ATTACK_DYNAMIC_DIM),
            torch.zeros(HISTORY_STEPS + 1, dtype=torch.long),
            categorical,
            torch.zeros(1, OPTION_NUMERIC_DIM),
            torch.zeros(1, POKEMON_DYNAMIC_DIM),
            torch.zeros(1, ATTACK_DYNAMIC_DIM),
            torch.tensor([0]),
            torch.tensor([0, 1]),
        )

        self.assertEqual(tuple(logits.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
