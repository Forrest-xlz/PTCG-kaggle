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
from beam_search.plot import aggregate_results, write_outputs
from beam_search.search import SearchDecision, SearchStats


def result(
    game_id: int,
    condition: str,
    opponent: str,
    outcome: str,
    *,
    error: str = "",
) -> GameResult:
    target_player = game_id % 2
    winner = {
        "win": target_player,
        "loss": 1 - target_player,
        "draw": 2,
        "failed": -1,
    }[outcome]
    return GameResult(
        game_id=game_id,
        condition=condition,
        target_deck="festival_lead",
        opponent_deck=opponent,
        target_player=target_player,
        player0_deck="festival_lead" if target_player == 0 else opponent,
        player1_deck=opponent if target_player == 0 else "festival_lead",
        outcome=outcome,
        winner=winner,
        duration_seconds=1.5,
        selections=10,
        search_calls=3 if condition == "beam" else 0,
        expanded_nodes=20 if condition == "beam" else 0,
        max_depth=4 if condition == "beam" else 0,
        truncated_trajectories=1 if condition == "beam" else 0,
        branch_errors=0,
        greedy_fallbacks=1 if condition == "beam" else 0,
        scored_steps=6 if condition == "beam" else 0,
        opponent_steps=2 if condition == "beam" else 0,
        warnings="",
        error=error,
    )


def sample_results() -> list[GameResult]:
    return [
        result(0, "beam", "a", "win"),
        result(1, "beam", "a", "win"),
        result(2, "beam", "a", "win"),
        result(3, "beam", "a", "loss"),
        result(4, "greedy", "a", "win"),
        result(5, "greedy", "a", "loss"),
        result(6, "greedy", "a", "loss"),
        result(7, "greedy", "a", "loss"),
        result(8, "beam", "c", "win"),
        result(9, "beam", "c", "draw"),
        result(10, "beam", "c", "failed", error="engine"),
        result(11, "greedy", "c", "win"),
        result(12, "greedy", "c", "loss"),
        result(13, "greedy", "c", "draw"),
    ]


def test_report_compares_conditions_from_target_perspective() -> None:
    report = aggregate_results(
        sample_results(), "festival_lead", ("a", "c")
    )

    against_a = next(
        row for row in report.comparisons if row["opponent_deck"] == "a"
    )
    assert against_a["beam_win_rate"] == pytest.approx(0.75)
    assert against_a["greedy_win_rate"] == pytest.approx(0.25)
    assert against_a["beam_uplift"] == pytest.approx(0.5)
    assert report.overall_beam["win_rate"] == pytest.approx(0.8)
    assert report.overall_greedy["win_rate"] == pytest.approx(2 / 6)
    assert report.overall_uplift == pytest.approx(0.8 - 2 / 6)
    against_c = next(
        row for row in report.comparisons if row["opponent_deck"] == "c"
    )
    assert against_c["beam_draws"] == 1
    assert against_c["beam_failures"] == 1
    assert against_c["beam_win_rate"] == pytest.approx(1.0)


def test_write_outputs_creates_focused_artifacts(tmp_path: Path) -> None:
    results = sample_results()
    report = aggregate_results(results, "festival_lead", ("a", "c"))

    write_outputs(tmp_path, results, report)

    expected = {
        "games.csv",
        "opponent_summary.csv",
        "festival_comparison.csv",
        "festival_win_rate_comparison.png",
        "summary.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    with (tmp_path / "festival_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["opponent_deck"] for row in rows] == ["a", "c"]
    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))
    assert summary["target_deck"] == "festival_lead"
    assert summary["overall_beam"]["wins"] == 4


class Backend:
    def __init__(self, observation, *, winner: int):
        self.observation_value = observation
        self.terminal = SimpleNamespace(
            current=SimpleNamespace(yourIndex=1, turn=2, result=winner),
            select=None,
        )
        self.selected: list[list[int]] = []
        self.finished = 0

    def start(self, deck0, deck1):
        return self.observation_value, SimpleNamespace(
            errorPlayer=-1, errorType=0
        )

    def select(self, action):
        self.selected.append(action)
        return self.terminal

    def finish(self):
        self.finished += 1

    def observation(self, value):
        return value


class Policy:
    def __init__(self):
        self.greedy_calls: list[tuple[int, bool]] = []
        self.recorded: list[list[int]] = []

    def record_action(self, observation, action, history):
        self.recorded.append(action)

    def greedy(self, observation, deck, history, *, record=False):
        self.greedy_calls.append((int(observation.current.yourIndex), record))
        return [0]


class Searcher:
    def __init__(self):
        self.calls = 0

    def choose(self, *args, **kwargs):
        self.calls += 1
        return SearchDecision([1], -0.2, SearchStats(search_calls=1))


def _play_one_action(
    *, condition: str, target_player: int, actor: int, turn: int, winner: int
):
    target = DeckSpec("festival_lead", tuple(range(60)))
    opponent = DeckSpec("opponent", tuple(reversed(range(60))))
    game = GameSpec(
        game_id=0,
        condition=condition,
        target_deck=target,
        opponent_deck=opponent,
        target_player=target_player,
        seed=9,
    )
    active = SimpleNamespace(
        current=SimpleNamespace(yourIndex=actor, turn=turn, result=-1),
        select=SimpleNamespace(),
    )
    backend = Backend(active, winner=winner)
    policy = Policy()
    searcher = Searcher()
    determinize_calls = []
    inputs = SearchInputs((), (), (), (), (), (), ())

    def determinize(*args, **kwargs):
        determinize_calls.append(args)
        return inputs

    actual = run_game(
        game,
        policy,
        searcher,
        frozenset(),
        backend=backend,
        determinize=determinize,
    )
    return actual, backend, policy, searcher, determinize_calls


def test_beam_search_runs_only_for_target_after_setup() -> None:
    actual, backend, policy, searcher, determinize_calls = _play_one_action(
        condition="beam", target_player=0, actor=0, turn=1, winner=0
    )

    assert actual.outcome == "win"
    assert backend.selected == [[1]]
    assert searcher.calls == 1
    assert len(determinize_calls) == 1
    assert policy.greedy_calls == []
    assert policy.recorded == [[1]]
    assert backend.finished == 1


@pytest.mark.parametrize(
    ("condition", "turn", "target_player", "actor"),
    [
        ("greedy", 1, 0, 0),
        ("beam", 0, 0, 0),
        ("beam", 1, 0, 1),
    ],
)
def test_non_beam_decisions_use_greedy_without_determinization(
    condition: str, turn: int, target_player: int, actor: int
) -> None:
    _, backend, policy, searcher, determinize_calls = _play_one_action(
        condition=condition,
        target_player=target_player,
        actor=actor,
        turn=turn,
        winner=target_player,
    )

    assert backend.selected == [[0]]
    assert searcher.calls == 0
    assert determinize_calls == []
    assert policy.greedy_calls == [(actor, True)]


@pytest.mark.parametrize(
    ("target_player", "winner", "expected"),
    [(0, 0, "win"), (1, 1, "win"), (0, 1, "loss"), (1, 0, "loss"), (1, 2, "draw")],
)
def test_outcome_is_always_from_target_perspective(
    target_player: int, winner: int, expected: str
) -> None:
    actual, *_ = _play_one_action(
        condition="greedy",
        target_player=target_player,
        actor=target_player,
        turn=1,
        winner=winner,
    )

    assert actual.outcome == expected
