from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extraction.replay_timing import (
    PLAYER_COLUMNS,
    ExtractionSettings,
    extract_archive,
    load_settings,
    resolve_archive,
)


def _payload(episode_id: str, first_team: str = "Alpha") -> dict:
    return {
        "info": {"EpisodeId": episode_id, "TeamNames": [first_team, "Beta"]},
        "steps": [
            [
                {"observation": {"remainingOverageTime": 600.0}},
                {"observation": {"remainingOverageTime": 600.0}},
            ],
            [
                {"observation": {"remainingOverageTime": 598.0}},
                {"observation": {"remainingOverageTime": 599.0}},
            ],
            [
                {"observation": {"remainingOverageTime": 597.5}},
                {"observation": {"remainingOverageTime": 598.75}},
            ],
        ],
    }


def _write_archive(path: Path, *, malformed_members: int = 0) -> Path:
    manifest = (
        "episode_id,avg_score,min_score,sum_score,agent_count\n"
        "good,1200,1100,2400,2\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.csv", manifest)
        archive.writestr("good.json", json.dumps(_payload("good")))
        for index in range(malformed_members):
            archive.writestr(f"bad-{index}.json", "{not-json")
    return path


def test_settings_have_only_supported_controls(tmp_path: Path) -> None:
    config = tmp_path / "extract.yaml"
    config.write_text(
        "extract:\n"
        "  input: archives\n"
        "  output: timing\n"
        "  date: latest\n"
        "  force: false\n",
        encoding="utf-8",
    )

    settings = load_settings(config, project_root=tmp_path)

    assert settings == ExtractionSettings(
        input=(tmp_path / "archives").resolve(),
        output=(tmp_path / "timing").resolve(),
        date="latest",
        force=False,
    )

    config.write_text(
        "extract:\n"
        "  input: archives\n"
        "  output: timing\n"
        "  date: latest\n"
        "  force: false\n"
        "  progress_every: 500\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported extract settings"):
        load_settings(config, project_root=tmp_path)


def test_resolve_archive_uses_parsed_latest_and_explicit_date(tmp_path: Path) -> None:
    for name in ("8.9.zip", "8.15.zip", "7.31.zip"):
        (tmp_path / name).touch()

    assert resolve_archive(tmp_path, "latest") == ("8.15", tmp_path / "8.15.zip")
    assert resolve_archive(tmp_path, "8.9") == ("8.9", tmp_path / "8.9.zip")
    with pytest.raises(FileNotFoundError, match="8.10"):
        resolve_archive(tmp_path, "8.10")


def test_extract_archive_writes_player_rows_and_reuses_valid_cache(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "8.15.zip")
    output = tmp_path / "out"

    summary = extract_archive(archive, output, "8.15", force=False)
    rows = pd.read_csv(output / "8.15.player_timings.csv")

    assert summary == {"status": "written", "player_rows": 2, "failed_members": 0}
    assert list(rows.columns) == list(PLAYER_COLUMNS)
    assert rows.loc[0, "startup_time_seconds"] == 2.0
    assert rows.loc[0, "mean_step_time_seconds"] == 0.5
    assert rows.loc[0, "max_score"] == 1300.0
    assert extract_archive(archive, output, "8.15", force=False)["status"] == "skipped"


def test_all_errors_are_written_and_clean_run_removes_stale_error_file(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "8.15.zip", malformed_members=2)
    output = tmp_path / "out"

    summary = extract_archive(archive, output, "8.15", force=True)
    errors = pd.read_csv(output / "8.15.extraction_errors.csv")

    assert summary["failed_members"] == 2
    assert len(errors) == 2

    _write_archive(archive, malformed_members=0)
    extract_archive(archive, output, "8.15", force=True)
    assert not (output / "8.15.extraction_errors.csv").exists()
