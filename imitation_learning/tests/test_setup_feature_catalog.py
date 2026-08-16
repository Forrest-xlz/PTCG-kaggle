from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.features import (  # noqa: E402
    _attack_energy_deficit,
    _build_setup_feature_catalog,
)


def skill(text: str):
    return SimpleNamespace(name="rule", text=text)


def card(
    card_id: int,
    name: str,
    *,
    card_type: int = 0,
    energy_type: int = 0,
    basic: bool = False,
    stage1: bool = False,
    stage2: bool = False,
    evolves_from: str | None = None,
    attacks: tuple[int, ...] = (),
    skills: tuple[object, ...] = (),
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


class SetupFeatureCatalogTests(unittest.TestCase):
    def test_builds_card_and_attack_aligned_metadata(self):
        cards = [
            card(0, "Seed", basic=True, attacks=(0,)),
            card(1, "Bloom", stage1=True, evolves_from="Seed", attacks=(1,)),
            card(2, "Basic Grass", card_type=5, energy_type=1),
            card(
                3,
                "Growth Stadium",
                card_type=4,
                skills=(skill(
                    "Each player's {G} Pokemon can evolve into {G} Pokemon "
                    "during the turn they play those Pokemon."
                ),),
            ),
        ]
        attacks = [
            SimpleNamespace(attackId=0, energies=[1]),
            SimpleNamespace(attackId=1, energies=[1, 0]),
        ]

        catalog = _build_setup_feature_catalog(cards, attacks, card_count=5)

        self.assertEqual(catalog.card_names, ("Seed", "Bloom", "Basic Grass", "Growth Stadium", ""))
        self.assertEqual(catalog.card_stages.tolist(), [0, 1, -1, -1, -1])
        self.assertEqual(catalog.evolves_from, (None, "Seed", None, None, None))
        self.assertEqual(catalog.card_types.tolist(), [0, 0, 5, 4, -1])
        self.assertEqual(catalog.energy_types.tolist(), [0, 0, 1, 0, -1])
        self.assertEqual(catalog.basic_energy.tolist(), [False, False, True, False, False])
        self.assertEqual(catalog.attack_costs, ((1,), (1, 0)))
        self.assertEqual(catalog.same_turn_evolution_types[3], (1,))

    def test_matches_special_and_colorless_energy_units(self):
        self.assertEqual(_attack_energy_deficit([1, 2], [1, 0]), 0)
        self.assertEqual(_attack_energy_deficit([2], [1]), 1)
        self.assertEqual(_attack_energy_deficit([10], [4]), 0)
        self.assertEqual(_attack_energy_deficit([11], [5]), 0)
        self.assertEqual(_attack_energy_deficit([11], [7]), 0)
        self.assertEqual(_attack_energy_deficit([11], [1]), 1)
        self.assertEqual(_attack_energy_deficit([1, 2, 3], [0, 0]), 0)
        self.assertEqual(_attack_energy_deficit([1], [1, 0]), 1)
        self.assertEqual(_attack_energy_deficit([10, 1], [1, 2]), 0)


if __name__ == "__main__":
    unittest.main()
