from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CG_ROOT = ROOT.parent / "pokemon_tcg_ai_battle" / "sample_submission"
if str(CG_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_ROOT))


from cg.api import EnergyType  # noqa: E402
from model.card_features import CARD_FEATURE_DIM  # noqa: E402
from model.features import (  # noqa: E402
    ENCODER_POKEMON_DYNAMIC_DIM,
    OPPONENT_SUMMARY_DIM,
    OWN_SUMMARY_DIM,
    PLAYER_SUMMARY_DIM,
    NumericFeatureCatalog,
    _attack_energy_readiness,
    _player_summary,
    encoder_pokemon_dynamic_features,
)


def _catalog() -> NumericFeatureCatalog:
    card_attacks = [()] * 8
    card_attacks[1] = (0, 1)
    return NumericFeatureCatalog(
        card_features=np.zeros((8, CARD_FEATURE_DIM), dtype=np.float32),
        attack_damage=np.asarray([0.2, 0.4], dtype=np.float32),
        card_attacks=tuple(card_attacks),
        attack_energies=(
            (int(EnergyType.FIRE), int(EnergyType.COLORLESS)),
            (int(EnergyType.DARKNESS),),
        ),
    )


def _pokemon(**overrides):
    values = dict(
        id=1,
        hp=90,
        maxHp=120,
        tools=[SimpleNamespace(id=3)],
        energyCards=[SimpleNamespace(id=4), SimpleNamespace(id=5)],
        energies=[EnergyType.FIRE, EnergyType.WATER],
        preEvolution=[SimpleNamespace(id=6), SimpleNamespace(id=7)],
        appearThisTurn=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _player(active=None, bench=None, bench_max=8):
    return SimpleNamespace(
        active=[] if active is None else [active],
        bench=list(bench or []),
        benchMax=bench_max,
        deckCount=40,
        handCount=5,
        hand=[],
        discard=[],
        prize=[None] * 6,
        poisoned=True,
        burned=False,
        asleep=True,
        paralyzed=False,
        confused=True,
    )


def test_energy_readiness_matches_special_and_colorless_energy() -> None:
    assert _attack_energy_readiness(
        [EnergyType.FIRE, EnergyType.COLORLESS],
        [EnergyType.FIRE, EnergyType.WATER],
    ) == 0
    assert _attack_energy_readiness(
        [EnergyType.GRASS], [EnergyType.RAINBOW]
    ) == 0
    assert _attack_energy_readiness(
        [EnergyType.PSYCHIC], [EnergyType.TEAM_ROCKET]
    ) == 0
    assert _attack_energy_readiness(
        [EnergyType.DARKNESS], [EnergyType.TEAM_ROCKET]
    ) == 0
    assert _attack_energy_readiness(
        [EnergyType.PSYCHIC, EnergyType.DARKNESS],
        [EnergyType.TEAM_ROCKET],
    ) == 1
    assert _attack_energy_readiness(
        [EnergyType.COLORLESS], [EnergyType.METAL]
    ) == 0


def test_encoder_dynamic_features_include_status_and_two_attacks() -> None:
    pokemon = _pokemon()
    features = encoder_pokemon_dynamic_features(
        pokemon,
        _player(active=pokemon),
        is_active=True,
        is_own=True,
        catalog=_catalog(),
    )

    assert features.shape == (ENCODER_POKEMON_DYNAMIC_DIM,)
    assert ENCODER_POKEMON_DYNAMIC_DIM == 38
    np.testing.assert_array_equal(features[23:28], [1, 0, 1, 0, 1])
    np.testing.assert_allclose(features[28:33], [1, 0.2, 0.4, 0, 1])
    np.testing.assert_allclose(features[33:38], [1, 0.4, 0.2, 0.2, 0])


def test_bench_has_no_active_special_conditions() -> None:
    pokemon = _pokemon()
    features = encoder_pokemon_dynamic_features(
        pokemon,
        _player(bench=[pokemon]),
        is_active=False,
        is_own=False,
        catalog=_catalog(),
    )
    np.testing.assert_array_equal(features[23:28], np.zeros(5))


def test_player_summary_appends_capacity_energy_hp_and_field_totals() -> None:
    active = _pokemon(hp=90, maxHp=120, energies=[EnergyType.FIRE] * 2)
    bench_a = _pokemon(hp=40, maxHp=100, energies=[EnergyType.WATER] * 3)
    bench_b = _pokemon(hp=70, maxHp=140, energies=[EnergyType.DARKNESS])
    summary = _player_summary(
        _player(active=active, bench=[bench_a, bench_b], bench_max=7),
        _catalog(),
    )

    assert len(summary) == PLAYER_SUMMARY_DIM == 79
    tail = summary[-25:]
    assert tail[0] == 7 / 8
    assert tail[1] == 2 / 7
    assert tail[2] == 2 / 10
    assert tail[3:7] == [100 / 400, 3 / 10, 140 / 400, 1 / 10]
    assert tail[19] == 4 / 80
    assert tail[20] == 40 / 400
    assert tail[21] == 6 / 90
    assert tail[22] == 6 / 90
    assert tail[23] == 200 / 3600
    assert tail[24] == 360 / 3600
    assert OWN_SUMMARY_DIM == 94
    assert OPPONENT_SUMMARY_DIM == 96


def test_empty_bench_minimum_hp_and_ratio_are_zero() -> None:
    summary = _player_summary(_player(active=_pokemon(), bench=[]), _catalog())
    tail = summary[-25:]
    assert tail[1] == 0
    assert tail[20] == 0
