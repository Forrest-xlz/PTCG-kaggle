from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from eda.deck_trends_eda import (
    CONFIG_PATH,
    FIGURE_FILENAMES,
    TABLE_FILENAMES,
    _chart_settings,
    _publish_artifacts,
)


def test_eda_config_defines_three_independent_charts_without_filenames() -> None:
    analysis = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["analysis"]
    line = _chart_settings(analysis, "line_chart", mirrors=True)
    sankey = _chart_settings(analysis, "sankey", mirrors=False)
    matrix = _chart_settings(analysis, "matchup_matrix", mirrors=True)
    assert line["score_filter"].mode == "all"
    assert sankey["score_filter"].mode == "min"
    assert matrix["score_filter"].mode == "min"
    assert all("filename" not in settings for settings in (line, sankey, matrix))


def test_publish_replaces_only_fixed_outputs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    (output / "tables").mkdir(parents=True)
    (staging / "tables").mkdir(parents=True)
    unrelated = output / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    for filename in FIGURE_FILENAMES.values():
        (output / filename).write_bytes(b"old")
        (staging / filename).write_bytes(b"new")
    for filename in TABLE_FILENAMES:
        (output / "tables" / filename).write_bytes(b"old")
        (staging / "tables" / filename).write_bytes(b"new")

    _publish_artifacts(staging, output)

    assert all((output / name).read_bytes() == b"new" for name in FIGURE_FILENAMES.values())
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_missing_staged_artifact_preserves_previous_outputs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    (output / "tables").mkdir(parents=True)
    (staging / "tables").mkdir(parents=True)
    for filename in FIGURE_FILENAMES.values():
        (output / filename).write_bytes(b"old")
        (staging / filename).write_bytes(b"new")
    for filename in TABLE_FILENAMES:
        (output / "tables" / filename).write_bytes(b"old")
        (staging / "tables" / filename).write_bytes(b"new")
    (staging / next(iter(FIGURE_FILENAMES.values()))).unlink()

    with pytest.raises(FileNotFoundError, match="Missing staged"):
        _publish_artifacts(staging, output)

    assert all((output / name).read_bytes() == b"old" for name in FIGURE_FILENAMES.values())


def test_pure_python_eda_replaces_notebook() -> None:
    assert (PROJECT_ROOT / "eda" / "deck_trends_eda.py").is_file()
    assert not (PROJECT_ROOT / "notebooks" / "deck_trends.ipynb").exists()
    assert (PROJECT_ROOT / "cfg" / "extract_deck_trend_data.yaml").is_file()
    assert (PROJECT_ROOT / "cfg" / "deck_trends_eda.yaml").is_file()
