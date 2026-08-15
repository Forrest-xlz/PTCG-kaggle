from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deck_strength import runner
from deck_strength.battle import GameResult
from deck_strength.config import (
    DeckSpec,
    DeckStrengthSettings,
    GameSpec,
    RuntimeSettings,
)
from deck_strength.evaluate import format_progress


def _settings(*, workers: int) -> DeckStrengthSettings:
    decks = (
        DeckSpec("a", tuple(range(60))),
        DeckSpec("b", tuple(reversed(range(60)))),
    )
    return DeckStrengthSettings(
        cg_path=Path("cg"),
        checkpoint=Path("model.pt"),
        device="cpu",
        seed=7,
        games_per_pair=3,
        output=Path("output"),
        runtime=RuntimeSettings(workers, 1),
        decks=decks,
    )


def _games(count: int = 3) -> tuple[GameSpec, ...]:
    settings = _settings(workers=1)
    deck_a, deck_b = settings.decks
    return tuple(
        GameSpec(index, deck_a, deck_b, index % 2, 10 + index)
        for index in range(count)
    )


def _result(game_id: int) -> GameResult:
    return GameResult(
        game_id=game_id,
        deck_a="a",
        deck_b="b",
        deck_a_player=game_id % 2,
        player0_deck="a" if game_id % 2 == 0 else "b",
        player1_deck="b" if game_id % 2 == 0 else "a",
        winner=game_id % 2,
        outcome_a="win",
        outcome_b="loss",
        duration_seconds=0.1,
        selections=2,
        error="",
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.game_ids: list[int] = []

    def run(self, game: GameSpec) -> GameResult:
        self.game_ids.append(game.game_id)
        return _result(game.game_id)


def test_worker_initialization_is_reused() -> None:
    settings = _settings(workers=1)
    runtime = FakeRuntime()
    with patch.object(runner, "build_worker_runtime", return_value=runtime) as build:
        runner.initialize_worker(settings)
        first = runner.run_worker_game(_games()[0])
        second = runner.run_worker_game(_games()[1])

    build.assert_called_once_with(settings)
    assert [first.game_id, second.game_id] == [0, 1]
    assert runtime.game_ids == [0, 1]


def test_execute_games_sequential_reports_every_result() -> None:
    settings = _settings(workers=1)
    runtime = FakeRuntime()
    seen: list[GameResult] = []
    with patch.object(runner, "build_worker_runtime", return_value=runtime):
        actual = runner.execute_games(settings, _games(), on_result=seen.append)

    assert [item.game_id for item in actual] == [0, 1, 2]
    assert seen == actual
    assert runtime.game_ids == [0, 1, 2]


def test_bounded_collection_returns_results_sorted_by_game_id() -> None:
    games = tuple(reversed(_games()))
    seen: list[GameResult] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = runner.collect_bounded_results(
            executor,
            games,
            lambda game: _result(game.game_id),
            max_pending=2,
            on_result=seen.append,
        )

    assert [item.game_id for item in actual] == [0, 1, 2]
    assert {item.game_id for item in seen} == {0, 1, 2}


def test_worker_game_requires_initializer() -> None:
    runner.clear_worker_runtime()
    try:
        runner.run_worker_game(_games(1)[0])
    except RuntimeError as exc:
        assert "not initialized" in str(exc)
    else:
        raise AssertionError("run_worker_game should require initialization")


def test_format_progress_contains_game_and_seat_information() -> None:
    line = format_progress(3, 12, _result(8))

    assert "[3/12]" in line
    assert "game_id=8" in line
    assert "a vs b" in line
    assert "a_player=0" in line
    assert "outcome_a=win" in line
    assert "selections=2" in line
