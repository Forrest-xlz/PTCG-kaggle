from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.deck_trends_eda import CONFIG_PATH, _chart_settings


def test_eda_config_defines_three_independent_charts() -> None:
    analysis = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["analysis"]
    line = _chart_settings(analysis, "line_chart", mirrors=True)
    sankey = _chart_settings(analysis, "sankey", mirrors=False)
    matrix = _chart_settings(analysis, "matchup_matrix", mirrors=True)
    assert line["score_filter"].mode == "all"
    assert sankey["score_filter"].mode == "min"
    assert matrix["score_filter"].mode == "min"
    assert len({line["filename"], sankey["filename"], matrix["filename"]}) == 3


def test_pure_python_eda_replaces_notebook() -> None:
    assert (PROJECT_ROOT / "notebooks" / "deck_trends_eda.py").is_file()
    assert not (PROJECT_ROOT / "notebooks" / "deck_trends.ipynb").exists()
    assert (PROJECT_ROOT / "cfg" / "extract_deck_trend_data.yaml").is_file()
    assert (PROJECT_ROOT / "cfg" / "deck_trends_eda.yaml").is_file()
