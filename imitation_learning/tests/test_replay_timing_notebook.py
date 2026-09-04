from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.replay_timing_eda import (
    CONFIG_PATH,
    ClusteringSettings,
    ScoreFilter,
    aggregate_team_timings,
    assign_timing_clusters,
    filter_team_timings,
    resolve_timing_csv,
)
import notebooks.replay_timing_eda as replay_timing_eda


def _player_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "8.15", "episode_id": "1", "player_index": 0, "team_name": "Alpha", "startup_time_seconds": 2.0, "subsequent_time_seconds": 3.0, "subsequent_action_count": 2, "mean_step_time_seconds": 1.5, "avg_score": 1100.0, "min_score": 1000.0, "max_score": 1200.0, "sum_score": 2200.0},
            {"date": "8.15", "episode_id": "2", "player_index": 0, "team_name": "Alpha", "startup_time_seconds": 4.0, "subsequent_time_seconds": 2.0, "subsequent_action_count": 1, "mean_step_time_seconds": 2.0, "avg_score": 1200.0, "min_score": 1100.0, "max_score": 1300.0, "sum_score": 2400.0},
            {"date": "8.15", "episode_id": "1", "player_index": 1, "team_name": "Beta", "startup_time_seconds": 1.0, "subsequent_time_seconds": 1.0, "subsequent_action_count": 2, "mean_step_time_seconds": 0.5, "avg_score": 1100.0, "min_score": 1000.0, "max_score": 1200.0, "sum_score": 2200.0},
        ]
    )


def test_replay_timing_eda_config_is_analysis_only() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["analysis"]
    assert set(raw) == {"input", "output", "date", "score_filter", "clustering"}
    assert set(raw["score_filter"]) == {"mode", "threshold"}
    assert set(raw["clustering"]) == {"n_clusters", "random_state", "n_init"}


def test_resolve_timing_csv_handles_latest_and_explicit_dates(tmp_path: Path) -> None:
    for name in ("8.9.player_timings.csv", "8.15.player_timings.csv"):
        (tmp_path / name).touch()
    assert resolve_timing_csv(tmp_path, "latest")[0] == "8.15"
    assert resolve_timing_csv(tmp_path, "8.9")[0] == "8.9"
    with pytest.raises(FileNotFoundError, match="8.10"):
        resolve_timing_csv(tmp_path, "8.10")


def test_team_aggregation_preserves_timing_and_score_metrics() -> None:
    teams = aggregate_team_timings(_player_rows())
    alpha = teams.set_index("team_name").loc["Alpha"]
    assert alpha["replay_count"] == 2
    assert alpha["startup_time_mean_seconds"] == 3.0
    assert alpha["subsequent_time_seconds"] == 5.0
    assert alpha["mean_step_time_seconds"] == pytest.approx(5 / 3)
    assert alpha["avg_score"] == 1150.0
    assert alpha["min_score"] == 1050.0
    assert alpha["max_score"] == 1250.0


@pytest.mark.parametrize(
    ("mode", "equal_value"),
    [("min", 1000.0), ("max", 1200.0), ("avg", 1100.0)],
)
def test_score_filter_is_inclusive(mode: str, equal_value: float) -> None:
    teams = aggregate_team_timings(_player_rows())
    filtered = filter_team_timings(teams, ScoreFilter(mode=mode, threshold=equal_value))
    assert "Beta" in filtered["team_name"].tolist()
    beta = filtered.set_index("team_name").loc["Beta"]
    assert beta["score_value"] == equal_value


def test_filtered_clusters_use_labels_from_global_cluster_space() -> None:
    teams = pd.DataFrame(
        {
            "team_name": ["A", "B", "C", "D"],
            "startup_time_mean_seconds": [1.0, 1.2, 8.0, 9.0],
            "mean_step_time_seconds": [0.1, 0.2, 2.0, 2.2],
        }
    )
    filtered = teams.loc[teams["team_name"].isin(["A", "D"])].copy()

    all_result, filtered_result, summary = assign_timing_clusters(
        teams, filtered, ClusteringSettings(n_clusters=2, random_state=42, n_init=10)
    )

    global_labels = all_result.set_index("team_name")["cluster"]
    filtered_labels = filtered_result.set_index("team_name")["cluster"]
    assert filtered_labels.to_dict() == {"A": global_labels["A"], "D": global_labels["D"]}
    assert summary["all_team_count"].sum() == 4
    assert summary["filtered_team_count"].sum() == 2


def _analysis_settings(tmp_path: Path):
    rows = []
    for index, (team, startup, action) in enumerate(
        [("A", 1.0, 0.1), ("B", 1.2, 0.2), ("C", 8.0, 2.0), ("D", 9.0, 2.2)]
    ):
        rows.append(
            {
                "date": "8.15",
                "episode_id": str(index),
                "player_index": 0,
                "team_name": team,
                "startup_time_seconds": startup,
                "subsequent_time_seconds": action * 2,
                "subsequent_action_count": 2,
                "mean_step_time_seconds": action,
                "avg_score": 1200.0,
                "min_score": 1100.0,
                "max_score": 1300.0,
                "sum_score": 2400.0,
            }
        )
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame(rows).to_csv(input_dir / "8.15.player_timings.csv", index=False)
    return replay_timing_eda.AnalysisSettings(
        input=input_dir,
        output=tmp_path / "output",
        date="latest",
        score_filter=ScoreFilter("min", 1100.0),
        clustering=ClusteringSettings(2, 42, 10),
    )


def test_run_writes_six_fixed_figures_and_three_tables(tmp_path: Path) -> None:
    settings = _analysis_settings(tmp_path)

    resolved_date, output = replay_timing_eda.run(settings)

    assert resolved_date == "8.15"
    assert {path.name for path in output.glob("*.png")} == set(replay_timing_eda.FIGURE_FILENAMES)
    assert {path.name for path in (output / "tables").glob("*.csv")} == set(replay_timing_eda.TABLE_FILENAMES)
    assert "cluster" in pd.read_csv(output / "tables" / "all_team_timings.csv")
    assert "cluster" in pd.read_csv(output / "tables" / "score_filtered_team_timings.csv")


def test_publish_replaces_owned_outputs_and_preserves_unrelated_file(tmp_path: Path) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    (output / "tables").mkdir(parents=True)
    (staging / "tables").mkdir(parents=True)
    unrelated = output / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    for filename in replay_timing_eda.FIGURE_FILENAMES:
        (output / filename).write_bytes(b"old")
        (staging / filename).write_bytes(b"new")
    for filename in replay_timing_eda.TABLE_FILENAMES:
        (output / "tables" / filename).write_bytes(b"old")
        (staging / "tables" / filename).write_bytes(b"new")

    replay_timing_eda.publish_artifacts(staging, output)

    assert all((output / name).read_bytes() == b"new" for name in replay_timing_eda.FIGURE_FILENAMES)
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_render_failure_preserves_previous_owned_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _analysis_settings(tmp_path)
    replay_timing_eda.run(settings)
    previous = {
        path.relative_to(settings.output): path.read_bytes()
        for path in settings.output.rglob("*")
        if path.is_file()
    }

    def fail_render(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(replay_timing_eda, "_save_scatter", fail_render)
    with pytest.raises(RuntimeError, match="render failed"):
        replay_timing_eda.run(settings)

    current = {
        path.relative_to(settings.output): path.read_bytes()
        for path in settings.output.rglob("*")
        if path.is_file()
    }
    assert current == previous


def test_obsolete_replay_timing_notebook_is_removed() -> None:
    assert not (PROJECT_ROOT / "notebooks" / "replay_timing.ipynb").exists()


def test_staging_directory_is_created_beside_output_and_cleaned(tmp_path: Path) -> None:
    output = tmp_path / "results" / "replay_timing"
    output.parent.mkdir(parents=True)

    with replay_timing_eda._staging_directory(output) as staging:
        assert staging.parent == output.parent
        assert staging.is_dir()
        (staging / "artifact.txt").write_text("temporary", encoding="utf-8")

    assert not staging.exists()
