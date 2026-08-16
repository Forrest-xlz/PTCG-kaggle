from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.attack_features import ATTACK_FEATURE_DIM
from model.card_features import CARD_FEATURE_DIM
from model.network import (
    CARD_REGION_INDEX,
    ENCODER_TOKENS,
    ModelConfig,
    PTCGTransformer,
    _encoder_card_mappings,
)


class Scale(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2


def config(*, residual: bool = False, layers: int = 1) -> ModelConfig:
    return ModelConfig(
        card_count=10,
        attack_count=3,
        encoder_size=256,
        d_model=4,
        num_heads=2,
        d_feedforward=8,
        known_deck_token_mlp_layers=layers,
        region_token_mlp_residual=residual,
    )


class KnownDeckNetworkTest(unittest.TestCase):
    def test_known_deck_has_independent_card_region(self) -> None:
        cfg = config()
        card_ids, regions = _encoder_card_mappings(cfg)
        known_start = 4 * (2 + 3 * cfg.card_count) + 4 * cfg.card_count

        self.assertEqual(ENCODER_TOKENS, 27)
        self.assertEqual(card_ids[known_start:known_start + 10].tolist(), list(range(10)))
        self.assertTrue(
            torch.all(
                regions[known_start:known_start + 10]
                == CARD_REGION_INDEX["own_known_deck"]
            )
        )
        self.assertNotEqual(
            CARD_REGION_INDEX["own_known_deck"],
            CARD_REGION_INDEX["own_deck"],
        )

    def test_known_deck_mlp_respects_replacement_and_residual_modes(self) -> None:
        tokens = torch.ones((1, 27, 4))
        for residual, expected in ((False, 2.0), (True, 3.0)):
            cfg = config(residual=residual)
            model = PTCGTransformer(
                cfg,
                torch.zeros((cfg.card_count, CARD_FEATURE_DIM)),
                torch.zeros((cfg.attack_count, ATTACK_FEATURE_DIM)),
            )
            model.own_known_deck_token_mlp = Scale()
            result = model.apply_region_token_mlps(tokens)
            self.assertEqual(tuple(result.shape), (1, 27, 4))
            self.assertEqual(float(result[0, 24, 0]), expected)
            self.assertEqual(float(result[0, 23, 0]), 1.0)

    def test_known_deck_mlp_layers_must_be_nonnegative(self) -> None:
        with self.assertRaisesRegex(ValueError, "known_deck_token_mlp_layers"):
            config(layers=-1)


if __name__ == "__main__":
    unittest.main()
