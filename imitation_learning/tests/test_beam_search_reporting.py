from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beam_search.battle import GameResult, run_game
from beam_search.config import DeckSpec, GameSpec
from beam_search.determinization import SearchInputs
from beam_search.search import SearchDecision, SearchStats
from beam_search.plot import aggregate_results, write_outputs


def result(
    game_id: int,
    beam: str,
    greedy: str,
    outcome: str,
    *,
    error: str = "",
) -> GameResult:
    return GameResult(
        game_id=game_id,
        beam_deck=beam,
        greedy_deck=greedy,
        beam_player=game_id % 2,
        outcome=outcome,
        winner=-1 if outcome in {"draw", "failed"} else (game_id % 2),
        duration_seconds=1.5,
        selections=10,
        search_calls=3,
        expanded_nodes=20,
        max_depth=4,
        truncated_trajectories=1,
        branch_errors=0,
        greedy_fallbacks=1,
        scored_steps=6,
        opponent_steps=2,
        warnings="",
        error=error,
    )


def sample_results() -> list[GameResult]:
    return [
        result(0, "a", "b", "win"),
        result(1, "a", "b", "win"),
        result(2, "a", "b", "win"),
        result(3, "a", "b", "loss"),
        result(4, "b", "a", "draw"),
        result(5, "b", "a", "failed", error="engine"),
        result(6, "a", "a", "win"),
        result(7, "b", "b", "loss"),
    ]


def test_matrix_uses_beam_rows_and_greedy_columns() -> None:
    report = aggregate_results(sample_results(), ("a", "b"))

    assert report.matrix[("a", "b")] == pytest.approx(0.75)
    assert report.matrix[("b", "a")] is None
    cell = next(
        row
        for row in report.matchups
        if row["beam_deck"] == "a" and row["greedy_deck"] == "b"
    )
    assert cell["attempted_games"] == 4
    assert cell["completed_games"] == 4
    assert cell["beam_wins"] == 3
    assert cell["beam_losses"] == 1
    assert cell["beam_win_rate"] == pytest.approx(0.75)
    assert report.overall["beam_wins"] == 4
    assert report.overall["beam_losses"] == 2
    assert report.overall["draws"] == 1
    assert report.overall["failures"] == 1


def test_write_outputs_creates_all_artifacts(tmp_path: Path) -> None:
    results = sample_results()
    report = aggregate_results(results, ("a", "b"))

    write_outputs(tmp_path, results, report, ("a", "b"))

    expected = {
        "games.csv",
        "matchup_summary.csv",
        "beam_win_rate_matrix.csv",
        "beam_win_rate_matrix.png",
        "summary.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    with (tmp_path / "beam_win_rate_matrix.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["beam_deck", "a", "b"]
    assert rows[1][0] == "a"
    assert float(rows[1][2]) == pytest.approx(0.75)
    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))
    assert summary["overall"]["beam_wins"] == 4


def test_run_game_uses_beam_for_assigned_player_and_finishes_engine() -> None:
    deck = DeckSpec("deck", tuple(range(60)))
    game = GameSpec(0, deck, deck, beam_player=0, seed=9)
    active = SimpleNamespace(
        current=SimpleNamespace(yourIndex=0, turn=1, result=-1),
        select=SimpleNamespace(),
    )
    terminal = SimpleNamespace(
        current=SimpleNamespace(yourIndex=1, turn=2, result=0),
        select=None,
    )

    class Backend:
        def __init__(self):
            self.selected = []
            self.finished = 0

        def start(self, deck0, deck1):
            return active, SimpleNamespace(errorPlayer=-1, errorType=0)

        def select(self, action):
            self.selected.append(action)
            return terminal

        def finish(self):
            self.finished += 1

        def observation(self, value):
            return value

    class Policy:
        def record_action(self, observation, action, history):
            return None

        def greedy(self, observation, deck, history, *, record=False):
            return [0]

    class Searcher:
        def choose(self, *args, **kwargs):
            return SearchDecision([1], -0.2, SearchStats(search_calls=1))

    backend = Backend()
    inputs = SearchInputs((), (), (), (), (), (), ())
    actual = run_game(
        game,
        Policy(),
        Searcher(),
        frozenset(),
        backend=backend,
        determinize=lambda *args, **kwargs: inputs,
    )

    assert actual.outcome == "win"
    assert actual.winner == 0
    assert backend.selected == [[1]]
    assert backend.finished == 1
