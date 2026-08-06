from __future__ import annotations

import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AreaType(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12


class OptionType(IntEnum):
    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


api = ModuleType("cg.api")
api.AreaType = AreaType
api.OptionType = OptionType
cg = ModuleType("cg")
cg.api = api
sys.modules.setdefault("cg", cg)
sys.modules.setdefault("cg.api", api)


from model.card_features import CARD_FEATURE_DIM  # noqa: E402
from model.features import (  # noqa: E402
    ATTACK_DYNAMIC_DIM,
    HISTORY_STRUCTURAL_DIM,
    OPTION_CATEGORICAL_DIM,
    POKEMON_DYNAMIC_DIM,
    NumericFeatureCatalog,
    history_action_features,
)


def card(card_id: int) -> SimpleNamespace:
    return SimpleNamespace(id=card_id, serial=100 + card_id)


def pokemon(card_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=card_id,
        serial=100 + card_id,
        hp=100,
        maxHp=120,
        energies=[7],
        energyCards=[card(7)],
        tools=[],
        preEvolution=[],
        appearThisTurn=False,
    )


def option(option_type: OptionType, **values) -> SimpleNamespace:
    fields = {
        "type": option_type,
        "number": None,
        "area": None,
        "index": None,
        "playerIndex": None,
        "toolIndex": None,
        "energyIndex": None,
        "count": None,
        "inPlayArea": None,
        "inPlayIndex": None,
        "attackId": None,
        "cardId": None,
        "serial": None,
        "specialConditionType": None,
    }
    fields.update(values)
    return SimpleNamespace(**fields)


def observation(options: list[SimpleNamespace]) -> SimpleNamespace:
    players = [
        SimpleNamespace(
            active=[pokemon(1)],
            bench=[pokemon(2)],
            hand=[card(3)],
            discard=[],
            prize=[],
        ),
        SimpleNamespace(
            active=[pokemon(4)],
            bench=[pokemon(5)],
            hand=None,
            discard=[],
            prize=[],
        ),
    ]
    return SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0,
            players=players,
            stadium=[],
            looking=[],
        ),
        select=SimpleNamespace(
            type=2,
            context=0,
            option=options,
            deck=None,
        ),
    )


def catalog() -> NumericFeatureCatalog:
    return NumericFeatureCatalog(
        card_features=np.zeros((32, CARD_FEATURE_DIM), dtype=np.float32),
        attack_damage=np.zeros(16, dtype=np.float32),
        card_attacks=tuple(() for _ in range(32)),
    )


def test_history_keeps_only_selected_option_rows() -> None:
    obs = observation([
        option(OptionType.PLAY, index=0),
        option(OptionType.END),
        option(OptionType.ATTACK, attackId=3),
    ])

    result = history_action_features(
        obs,
        [0, 2],
        card_count=32,
        attack_count=16,
        numeric_catalog=catalog(),
    )

    assert result.select_type == 2
    assert result.select_context == 0
    assert result.option_categorical.shape == (2, OPTION_CATEGORICAL_DIM)
    assert result.structural.shape == (2, HISTORY_STRUCTURAL_DIM)
    assert result.pokemon_dynamic.shape == (2, POKEMON_DYNAMIC_DIM)
    assert result.attack_dynamic.shape == (2, ATTACK_DYNAMIC_DIM)
    assert result.option_categorical[:, 0].tolist() == [7, 13]


def test_structural_fields_normalize_play_and_attack_regions() -> None:
    obs = observation([
        option(OptionType.PLAY, index=0),
        option(OptionType.ATTACK, attackId=3),
    ])

    result = history_action_features(
        obs,
        [0, 1],
        card_count=32,
        attack_count=16,
        numeric_catalog=catalog(),
    )

    play, attack = result.structural.tolist()
    assert play[:5] == [7, int(AreaType.HAND), 0, 1, 0]
    assert attack[:5] == [13, int(AreaType.ACTIVE), int(AreaType.ACTIVE), 1, 2]


def test_empty_action_retains_decision_and_has_no_option_rows() -> None:
    result = history_action_features(
        observation([option(OptionType.YES), option(OptionType.NO)]),
        [],
        card_count=32,
        attack_count=16,
        numeric_catalog=catalog(),
    )

    assert result.select_type == 2
    assert result.select_context == 0
    assert result.option_categorical.shape == (0, OPTION_CATEGORICAL_DIM)
    assert result.structural.shape == (0, HISTORY_STRUCTURAL_DIM)
    assert result.pokemon_dynamic.shape == (0, POKEMON_DYNAMIC_DIM)
    assert result.attack_dynamic.shape == (0, ATTACK_DYNAMIC_DIM)
