from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CG_ROOT = PROJECT_ROOT.parent / "pokemon_tcg_ai_battle" / "sample_submission"
if str(CG_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_ROOT))

from model import features as feature_module
from model.attack_features import ATTACK_FEATURE_DIM, build_attack_feature_table
from model.card_features import (
    CARD_FEATURE_DIM,
    CARD_RESISTANCE_OFFSET,
    CARD_TYPE_OFFSET,
    CARD_WEAKNESS_OFFSET,
)
from model.features import enumerate_actions


def test_actions_cover_every_count_from_max_to_min() -> None:
    assert enumerate_actions(3, min_count=0, max_count=2) == [
        [0, 1],
        [0, 2],
        [1, 2],
        [0],
        [1],
        [2],
        [],
    ]


def test_actions_respect_limit_without_materializing_all_combinations() -> None:
    assert enumerate_actions(100, min_count=1, max_count=2, limit=3) == [
        [0, 1],
        [0, 2],
        [0, 3],
    ]


@pytest.mark.parametrize(
    ("option_count", "min_count", "max_count"),
    [(2, -1, 1), (2, 2, 1), (2, 0, 3)],
)
def test_actions_reject_invalid_count_ranges(
    option_count: int, min_count: int, max_count: int
) -> None:
    with pytest.raises(ValueError, match="action counts"):
        enumerate_actions(option_count, min_count, max_count)


def _card(card_id: int):
    return SimpleNamespace(id=card_id, tools=[], energyCards=[])


def test_attack_static_table_matches_reference_contract() -> None:
    attack = SimpleNamespace(attackId=1, damage=150, energies=[0, 2, 2])

    table = build_attack_feature_table([attack], attack_count=3)

    assert table.shape == (3, ATTACK_FEATURE_DIM)
    assert table.dtype == torch.float32
    assert table[1, 0] == pytest.approx(0.5)
    assert table[1, 1] == 1
    assert table[1, 3] == 2
    assert table[1, -1] == pytest.approx(0.6)


def test_option_features_keep_entities_numeric_values_and_action_membership() -> None:
    card_features = np.zeros((8, CARD_FEATURE_DIM), dtype=np.float32)
    card_features[1, CARD_TYPE_OFFSET + 2] = 1
    card_features[2, CARD_WEAKNESS_OFFSET + 2] = 1
    card_features[2, CARD_RESISTANCE_OFFSET + 12] = 1
    card_features[3, CARD_TYPE_OFFSET + 4] = 1
    catalog = feature_module.NumericFeatureCatalog(
        card_features=card_features,
        attack_damage=np.asarray([0.0, 0.4, 0.8], dtype=np.float32),
        card_attacks=((),) * 8,
    )
    own = SimpleNamespace(
        hand=[_card(3)], discard=[], active=[_card(1)],
        bench=[_card(4)], prize=[],
    )
    opponent = SimpleNamespace(
        hand=[], discard=[], active=[_card(2)], bench=[], prize=[],
    )
    common = dict(
        number=None, playerIndex=None, toolIndex=None, energyIndex=None,
        count=None, cardId=None, specialConditionType=None,
    )
    options = [
        SimpleNamespace(
            type=8, area=2, index=0, inPlayArea=5, inPlayIndex=0,
            attackId=None, **common,
        ),
        SimpleNamespace(
            type=13, area=None, index=None, inPlayArea=None,
            inPlayIndex=None, attackId=1, **common,
        ),
    ]
    obs = SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0, players=[own, opponent], stadium=[], looking=[],
        ),
        select=SimpleNamespace(context=21, option=options, deck=[]),
    )

    encoded = feature_module.decoder_features(
        obs, [[0, 1], [0], []], 8, 3, numeric_catalog=catalog
    )

    np.testing.assert_array_equal(
        encoded.categorical,
        [[8, 21, 3, 4, 3], [13, 21, 8, 8, 1]],
    )
    assert encoded.numeric.shape == (2, 16)
    assert encoded.numeric[0, 10] == pytest.approx(0.5)
    assert encoded.numeric[0, 13] == pytest.approx(4 / 6)
    assert encoded.numeric[0, 14] == 1
    assert encoded.numeric[1, 12] == pytest.approx(0.4)
    np.testing.assert_array_equal(encoded.action_index, [0, 1, 0])
    np.testing.assert_array_equal(encoded.action_offset, [0, 2, 3, 3])
