from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beam_search import runner
from beam_search.battle import GameResult
from beam_search.config import (
    BeamSearchSettings,
    DeckSpec,
    GameSpec,
    RuntimeSettings,
    SearchSettings,
)


def _settings(*, workers: int) -> BeamSearchSettings:
    deck = DeckSpec("deck", tuple(range(60)))
    return BeamSearchSettings(
        cg_path=Path("cg"),
        checkpoint=Path("model.pt"),
        device="cpu",
        seed=7,
        games_per_matchup=3,
        output=Path("output"),
        runtime=RuntimeSettings(
            workers=workers,
            torch_threads_per_worker=1,
        ),
        search=SearchSettings(2, 2, 1.0, 8),
        decks=(deck,),
    )


def _games(count: int = 3) -> tuple[GameSpec, ...]:
    deck = DeckSpec("deck", tuple(range(60)))
    return tuple(GameSpec(index, deck, deck, index % 2, 10 + index) for index in range(count))


def _result(game_id: int) -> GameResult:
    return GameResult(
        game_id=game_id,
        beam_deck="deck",
        greedy_deck="deck",
        beam_player=game_id % 2,
        outcome="win",
        winner=game_id % 2,
        duration_seconds=0.1,
        selections=1,
        search_calls=1,
        expanded_nodes=1,
        max_depth=1,
        truncated_trajectories=0,
        branch_errors=0,
        greedy_fallbacks=0,
        scored_steps=1,
        opponent_steps=0,
        warnings="",
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
