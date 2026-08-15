from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deck_strength.battle import GameResult
from deck_strength.reporting import aggregate_results, write_outputs


def _result(
    game_id: int,
    deck_a: str,
    deck_b: str,
    outcome_a: str,
    *,
    error: str = "",
) -> GameResult:
    outcome_b = {
        "win": "loss",
        "loss": "win",
        "draw": "draw",
        "failed": "failed",
    }[outcome_a]
    return GameResult(
        game_id=game_id,
        deck_a=deck_a,
        deck_b=deck_b,
        deck_a_player=game_id % 2,
        player0_deck=deck_a if game_id % 2 == 0 else deck_b,
        player1_deck=deck_b if game_id % 2 == 0 else deck_a,
        winner=-1,
        outcome_a=outcome_a,
        outcome_b=outcome_b,
        duration_seconds=1.5,
        selections=10,
        error=error,
    )


def _sample_results() -> list[GameResult]:
    return [
        _result(0, "a", "b", "win"),
        _result(1, "a", "b", "loss"),
        _result(2, "a", "b", "draw"),
        _result(3, "a", "b", "failed", error="engine"),
        _result(4, "a", "c", "win"),
        _result(5, "b", "c", "win"),
    ]


def test_aggregate_builds_directed_matrix_and_ranking() -> None:
    report = aggregate_results(_sample_results(), ("a", "b", "c"))

    assert report.matrix[("a", "a")] is None
    assert report.matrix[("a", "b")] == pytest.approx(0.5)
    assert report.matrix[("b", "a")] == pytest.approx(0.5)
    assert report.completed[("a", "b")] == 3
    pair = next(row for row in report.matchups if row["deck_a"] == "a" and row["deck_b"] == "b")
    assert pair["attempted_games"] == 4
    assert pair["draws"] == 1
    assert pair["failures"] == 1
    assert [row["deck"] for row in report.ranking] == ["a", "b", "c"]
    assert report.ranking[0]["win_rate"] == pytest.approx(2 / 3)


def test_write_outputs_creates_all_artifacts(tmp_path: Path) -> None:
    names = ("a", "b", "c")
    results = _sample_results()
    report = aggregate_results(results, names)

    write_outputs(tmp_path, results, report, names)

    assert {path.name for path in tmp_path.iterdir()} == {
        "games.csv",
        "matchup_summary.csv",
        "win_rate_matrix.csv",
        "win_rate_matrix.png",
        "deck_ranking.csv",
        "summary.json",
    }
    with (tmp_path / "win_rate_matrix.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["deck", "a", "b", "c"]
    assert rows[1][0] == "a"
    assert rows[1][1] == ""
    assert float(rows[1][2]) == pytest.approx(0.5)
    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))
    assert summary["overall"]["attempted_games"] == 6
