from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.card_features import CARD_FEATURE_DIM  # noqa: E402
from model.features import (  # noqa: E402
    OPPONENT_SUMMARY_DIM as FEATURE_OPPONENT_DIM,
    OWN_SUMMARY_DIM as FEATURE_OWN_DIM,
)
from model.network import (  # noqa: E402
    ATTACK_FEATURE_DIM,
    ModelConfig,
    OPPONENT_SUMMARY_DIM as NETWORK_OPPONENT_DIM,
    OWN_SUMMARY_DIM as NETWORK_OWN_DIM,
    PTCGTransformer,
)
from training.feature_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    OPPONENT_SUMMARY_DIM as CACHE_OPPONENT_DIM,
    OWN_SUMMARY_DIM as CACHE_OWN_DIM,
)


class SetupSummaryIntegrationTests(unittest.TestCase):
    def test_widths_schema_and_projection_inputs_match(self):
        self.assertEqual(FEATURE_OWN_DIM, 86)
        self.assertEqual(FEATURE_OPPONENT_DIM, 84)
        self.assertEqual(NETWORK_OWN_DIM, 86)
        self.assertEqual(NETWORK_OPPONENT_DIM, 84)
        self.assertEqual(CACHE_OWN_DIM, 86)
        self.assertEqual(CACHE_OPPONENT_DIM, 84)
        self.assertEqual(CACHE_SCHEMA_VERSION, 19)

        config = ModelConfig(
            card_count=10,
            attack_count=3,
            encoder_size=256,
            d_model=8,
            num_heads=2,
            d_feedforward=16,
        )
        model = PTCGTransformer(
            config,
            torch.zeros((10, CARD_FEATURE_DIM)),
            torch.zeros((3, ATTACK_FEATURE_DIM)),
        )
        self.assertEqual(model.own_summary_projection.in_features, 86)
        self.assertEqual(model.opponent_summary_projection.in_features, 84)

    def test_all_signature_producers_use_setup_v5(self):
        expected = "numeric-summary-27-known-deck-setup-v5"
        for relative in (
            "training/cache_features.py",
            "training/train.py",
            "validation/evaluate.py",
        ):
            source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
