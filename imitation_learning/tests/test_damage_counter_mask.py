from __future__ import annotations

import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


class AreaType(IntEnum):
    BENCH = 5


cg = ModuleType("cg")
api = ModuleType("cg.api")
api.AreaType = AreaType
cg.api = api
sys.modules.setdefault("cg", cg)
sys.modules.setdefault("cg.api", api)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.features import damage_counter_action_eligibility


def _observation(context: int, bench_hps: list[int]) -> SimpleNamespace:
    opponent = SimpleNamespace(
        bench=[SimpleNamespace(hp=hp) for hp in bench_hps]
    )
    return SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0,
            players=[SimpleNamespace(bench=[]), opponent],
        ),
        select=SimpleNamespace(
            context=context,
            option=[
                SimpleNamespace(
                    area=AreaType.BENCH,
                    playerIndex=1,
                    index=index,
                )
                for index in range(len(bench_hps))
            ],
        ),
    )


def test_ko_target_is_ineligible_when_positive_hp_target_exists() -> None:
    result = damage_counter_action_eligibility(
        _observation(14, [0, 40]),
        [[0], [1]],
    )

    np.testing.assert_array_equal(result, [False, True])
    assert result.dtype == np.bool_


def test_all_ko_targets_are_restored_as_required_fallback() -> None:
    result = damage_counter_action_eligibility(
        _observation(14, [0, -10]),
        [[0], [1]],
    )

    np.testing.assert_array_equal(result, [True, True])


def test_combined_action_is_ineligible_if_any_target_is_ko() -> None:
    result = damage_counter_action_eligibility(
        _observation(14, [0, 40]),
        [[0, 1], [1]],
    )

    np.testing.assert_array_equal(result, [False, True])


def test_other_select_contexts_remain_unchanged() -> None:
    result = damage_counter_action_eligibility(
        _observation(13, [0, 40]),
        [[0], [1]],
    )

    np.testing.assert_array_equal(result, [True, True])


def test_unresolvable_option_remains_eligible() -> None:
    obs = _observation(14, [0, 40])
    obs.select.option[0].index = 99

    result = damage_counter_action_eligibility(obs, [[0], [1]])

    np.testing.assert_array_equal(result, [True, True])
