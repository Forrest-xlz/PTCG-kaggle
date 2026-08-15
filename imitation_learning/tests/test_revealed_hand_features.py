from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.card_features import CARD_FEATURE_DIM
from model.features import NumericFeatureCatalog, encoder_features


def _player(*, own_hand: bool) -> SimpleNamespace:
    return SimpleNamespace(
        active=[],
        bench=[],
        benchMax=8,
        deckCount=60,
        discard=[],
        prize=[None] * 6,
        handCount=0,
        hand=[] if own_hand else None,
        poisoned=False,
        burned=False,
        asleep=False,
        paralyzed=False,
        confused=False,
    )


def _observation() -> SimpleNamespace:
    state = SimpleNamespace(
        turn=1,
        turnActionCount=0,
        yourIndex=0,
        firstPlayer=0,
        supporterPlayed=False,
        stadiumPlayed=False,
        energyAttached=False,
        retreated=False,
        result=-1,
        stadium=[],
        looking=None,
        players=[_player(own_hand=True), _player(own_hand=False)],
    )
    select = SimpleNamespace(
        type=0,
        context=0,
        minCount=1,
        maxCount=1,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=[SimpleNamespace()],
    )
    return SimpleNamespace(current=state, select=select, logs=[])


class RevealedHandFeatureTests(unittest.TestCase):
    card_count = 1300
    catalog = NumericFeatureCatalog(
        card_features=np.zeros(
            (card_count, CARD_FEATURE_DIM), dtype=np.float32
        ),
        attack_damage=np.zeros(1, dtype=np.float32),
        card_attacks=tuple(() for _ in range(card_count)),
    )

    def encode(self, own=(), opponent=()):
        return encoder_features(
            _observation(),
            [],
            self.card_count,
            numeric_catalog=self.catalog,
            own_revealed_hand=own,
            opponent_revealed_hand=opponent,
        )

    def test_revealed_cards_use_two_new_sparse_words(self) -> None:
        encoded = self.encode([741, 741], [305])

        own_base = 8 + 17 * self.card_count
        opponent_base = own_base + self.card_count
        self.assertEqual(len(encoded.sparse.offset), 28)
        self.assertEqual(encoded.revealed_hand_present, [1, 1])
        self.assertEqual(
            encoded.sparse.index[-3:],
            [own_base + 741, own_base + 741, opponent_base + 305],
        )
        self.assertEqual(encoded.sparse.value[-3:], [1.0, 1.0, 1.0])

    def test_empty_revealed_hands_produce_masked_empty_words(self) -> None:
        encoded = self.encode()

        self.assertEqual(len(encoded.sparse.offset), 28)
        self.assertEqual(encoded.revealed_hand_present, [0, 0])
        self.assertEqual(
            encoded.sparse.offset[-2:],
            [len(encoded.sparse.index), len(encoded.sparse.index)],
        )


if __name__ == "__main__":
    unittest.main()
