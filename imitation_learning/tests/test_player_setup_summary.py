from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.card_features import CARD_FEATURE_DIM  # noqa: E402
from model.features import (  # noqa: E402
    NumericFeatureCatalog,
    _build_setup_feature_catalog,
    _own_hand_setup_summary,
    _public_setup_summary,
)


def static_card(
    card_id,
    name,
    *,
    card_type=0,
    energy_type=1,
    basic=False,
    stage1=False,
    stage2=False,
    evolves_from=None,
    attacks=(),
    skills=(),
):
    return SimpleNamespace(
        cardId=card_id,
        name=name,
        cardType=card_type,
        energyType=energy_type,
        basic=basic,
        stage1=stage1,
        stage2=stage2,
        evolvesFrom=evolves_from,
        attacks=list(attacks),
        skills=list(skills),
    )


STATIC_CARDS = [
    static_card(0, "Seed", basic=True, attacks=(0,)),
    static_card(1, "Bloom", stage1=True, evolves_from="Seed", attacks=(1,)),
    static_card(2, "Crown", stage2=True, evolves_from="Bloom", attacks=(2,)),
    static_card(3, "Solo", basic=True, attacks=(0,)),
    static_card(4, "Basic Grass", card_type=5, basic=False),
    static_card(
        5,
        "Growth Stadium",
        card_type=4,
        skills=(SimpleNamespace(
            text=(
                "Each player's {G} Pokemon can evolve into {G} Pokemon "
                "during the turn they play those Pokemon."
            )
        ),),
    ),
]
ATTACKS = [
    SimpleNamespace(attackId=0, energies=[1]),
    SimpleNamespace(attackId=1, energies=[1, 0]),
    SimpleNamespace(attackId=2, energies=[1, 1]),
]
DECK = [0, 1, 2, 3, 4, 5]


def catalog():
    return NumericFeatureCatalog(
        card_features=np.zeros((6, CARD_FEATURE_DIM), dtype=np.float32),
        attack_damage=np.zeros(3, dtype=np.float32),
        card_attacks=((0,), (1,), (2,), (0,), (), ()),
        setup=_build_setup_feature_catalog(STATIC_CARDS, ATTACKS, 6),
    )


def card(card_id):
    return SimpleNamespace(id=card_id)


def pokemon(card_id, *, energies=(), energy_cards=(), appeared=False):
    return SimpleNamespace(
        id=card_id,
        energies=list(energies),
        energyCards=[card(value) for value in energy_cards],
        appearThisTurn=appeared,
    )


def player(hand):
    return SimpleNamespace(
        active=[pokemon(0, energies=(1,), energy_cards=(4,))],
        bench=[
            pokemon(1, energy_cards=(4,), appeared=True),
            pokemon(2, energies=(1, 1), energy_cards=(4,)),
            pokemon(3),
        ],
        benchMax=5,
        hand=hand,
    )


class PlayerSetupSummaryTests(unittest.TestCase):
    def test_public_summary_describes_board_formation(self):
        values = _public_setup_summary(player(None), DECK, catalog())

        self.assertEqual(len(values), 11)
        self.assertAlmostEqual(values[0], 3 / 32)
        self.assertAlmostEqual(values[1], 3 / 64)
        self.assertEqual(values[2:5], [2 / 9, 1 / 9, 1 / 9])
        self.assertAlmostEqual(values[5], 2 / 9)
        self.assertAlmostEqual(values[6], 1 / 9)
        self.assertAlmostEqual(values[7], 2 / 9)
        self.assertAlmostEqual(values[8], 3 / 45)
        self.assertEqual(values[9], 0.0)
        self.assertAlmostEqual(values[10], 2 / 8)

    def test_own_hand_summary_respects_turn_and_same_turn_stadium(self):
        own = player([card(1), card(2), card(3), card(4)])
        state = SimpleNamespace(turn=2, stadium=[card(5)])

        values = _own_hand_setup_summary(state, own, DECK, catalog())

        self.assertEqual(len(values), 4)
        self.assertAlmostEqual(values[0], 2 / 9)
        self.assertAlmostEqual(values[1], 2 / 20)
        self.assertAlmostEqual(values[2], 1 / 20)
        self.assertAlmostEqual(values[3], 1 / 20)

        first_turn = SimpleNamespace(turn=1, stadium=[card(5)])
        self.assertEqual(
            _own_hand_setup_summary(first_turn, own, DECK, catalog())[0],
            0.0,
        )

    def test_public_opponent_summary_does_not_require_a_hand(self):
        opponent = player(None)
        values = _public_setup_summary(opponent, DECK, catalog())
        self.assertEqual(len(values), 11)


if __name__ == "__main__":
    unittest.main()
