from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.extract import _iter_player_records


def state(status: str, marker: int, action: list[int]) -> dict:
    return {
        "status": status,
        "action": action,
        "observation": {
            "current": {"marker": marker},
            "select": {"option": [], "minCount": 0, "maxCount": 0},
        },
    }


def test_action_is_taken_from_next_step_and_inactive_states_are_ignored() -> None:
    steps = [
        [state("ACTIVE", marker=10, action=[99])],
        [state("INACTIVE", marker=20, action=[2])],
        [state("ACTIVE", marker=30, action=[])],
        [state("DONE", marker=40, action=[4])],
    ]

    records = list(
        _iter_player_records(
            steps,
            player=0,
            episode=123,
            date="7.24",
            deck=[1] * 60,
        )
    )

    assert [(record["step"], record["selected"]) for record in records] == [
        (0, [2]),
        (2, [4]),
    ]
    assert [record["observation"]["current"]["marker"] for record in records] == [
        10,
        30,
    ]
