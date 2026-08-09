from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.expert_validation import load_expert_loser_date_info
from training.feature_cache import stable_episode_key


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["episode_id", "min_score", "sum_score", "agent_count"],
    )
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.csv", buffer.getvalue())


def row(episode_id: str, low: float, high: float) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "min_score": low,
        "sum_score": low + high,
        "agent_count": 2,
    }


def test_loser_cutoff_is_daily_and_requires_both_scores(tmp_path: Path) -> None:
    write_manifest(
        tmp_path / "7.23.zip",
        [row("a", 80, 100), row("b", 90, 90), row("c", 70, 95)],
    )
    write_manifest(
        tmp_path / "7.24.zip",
        [row("d", 180, 200), row("e", 190, 190), row("f", 170, 195)],
    )

    infos = load_expert_loser_date_info(
        tmp_path,
        required_dates=[(7, 23), (7, 24)],
        ratio=0.5,
    )

    assert infos[(7, 23)].cutoff == 90
    assert infos[(7, 23)].participant_count == 6
    assert infos[(7, 23)].episode_count == 3
    assert infos[(7, 23)].eligible_episode_keys == frozenset(
        {stable_episode_key("b")}
    )
    assert infos[(7, 24)].cutoff == 190
    assert infos[(7, 24)].eligible_episode_keys == frozenset(
        {stable_episode_key("e")}
    )


def test_loser_cutoff_includes_ties(tmp_path: Path) -> None:
    write_manifest(
        tmp_path / "7.24.zip",
        [row("a", 90, 100), row("b", 90, 100), row("c", 80, 100)],
    )

    info = load_expert_loser_date_info(
        tmp_path, required_dates=[(7, 24)], ratio=2 / 3
    )[(7, 24)]

    assert info.cutoff == 90
    assert info.eligible_episode_keys == frozenset(
        {stable_episode_key("a"), stable_episode_key("b")}
    )


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_loser_cutoff_rejects_invalid_ratio(tmp_path: Path, ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio"):
        load_expert_loser_date_info(
            tmp_path, required_dates=[], ratio=ratio
        )
