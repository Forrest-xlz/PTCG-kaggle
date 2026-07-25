from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
