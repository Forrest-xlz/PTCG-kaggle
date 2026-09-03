from __future__ import annotations

import csv
import gzip
import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extraction.deck_trend_data import (
    SCHEMA_VERSION,
    extract_team_names,
    load_settings,
    process_archive,
)


def write_config(path: Path, **overrides) -> Path:
    extract = {
        "input": "../replay_episodes",
        "output": "data/deck_trend",
        "workers": 4,
        "limit_members": None,
        "force": False,
        **overrides,
    }
    raw = {
        "extract": extract,
        "analysis": {
            "snapshot_interval_days": 3,
            "start_date": None,
            "end_date": None,
            "min_share_percent": 2.0,
            "exclude_mirror_matches": True,
        },
    }
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def replay_payload() -> dict:
    decks = [[1] * 60, [2] * 60]
    return {
        "info": {
            "EpisodeId": 123,
            "TeamNames": ["Alpha", "Beta"],
            "Agents": [{"Name": "A"}, {"Name": "B"}],
        },
        "rewards": [1, -1],
        "steps": [
            [
                {"visualize": [{"action": decks}]},
                {"visualize": []},
            ]
        ],
    }


def write_zip(path: Path, members: int = 1) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(members):
            archive.writestr(f"{index}.json", json.dumps(replay_payload()))
    return path


def test_load_settings_and_reject_invalid_values(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path / "trend.yaml"))
    assert settings.workers == 4
    assert settings.force is False
    for field, value in (("workers", 0), ("limit_members", 0), ("force", "no")):
        with pytest.raises(ValueError, match=field):
            load_settings(
                write_config(tmp_path / f"{field}.yaml", **{field: value})
            )


def test_team_names_prefer_team_names_and_fallback_to_agents() -> None:
    payload = replay_payload()
    assert extract_team_names(payload) == ["Alpha", "Beta"]
    payload["info"].pop("TeamNames")
    assert extract_team_names(payload) == ["A", "B"]
    payload["info"]["Agents"] = [{"Name": "?"}, {"Name": ""}]
    with pytest.raises(ValueError, match="team names"):
        extract_team_names(payload)


def test_archive_writes_two_player_rows_and_complete_metadata(
    tmp_path: Path,
) -> None:
    archive = write_zip(tmp_path / "7.1.zip")
    output = tmp_path / "out"
    summary = process_archive((str(archive), str(output), False, None))
    assert summary["status"] == "written"
    assert summary["complete"] is True
    assert summary["player_rows"] == 2

    with gzip.open(output / "7.1.players.csv.gz", "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert list(rows[0]) == [
        "date",
        "episode_id",
        "player",
        "team_name",
        "opponent_team_name",
        "deck",
        "reward",
        "result",
    ]
    assert rows[0]["team_name"] == "Alpha"
    assert rows[0]["opponent_team_name"] == "Beta"
    assert len(json.loads(rows[0]["deck"])) == 60

    meta = json.loads((output / "7.1.meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["complete"] is True
    assert process_archive((str(archive), str(output), False, None))["status"] == "skipped"


def test_limited_shard_is_not_reused_as_complete(tmp_path: Path) -> None:
    archive = write_zip(tmp_path / "7.2.zip", members=2)
    output = tmp_path / "out"
    limited = process_archive((str(archive), str(output), False, 1))
    assert limited["complete"] is False
    rebuilt = process_archive((str(archive), str(output), False, None))
    assert rebuilt["status"] == "written"
    assert rebuilt["processed"] == 2
    assert rebuilt["complete"] is True
