from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "eda" / "replay_timing.ipynb"


def test_score_filtered_team_details_cell_is_present() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "### Score-Filtered Team Details" in source
    assert 'mean_score=("score_value", "mean")' in source
    assert 'merge(score_summary, on="team_name", validate="one_to_one")' in source
    assert '"mean_startup_time_seconds"' in source
    assert '"mean_action_time_seconds"' in source
    assert 'sort_values(["mean_score", "team_name"]' in source
    assert "set(score_filtered_team_details[\"team_name\"])" in source
    assert "set(filtered_teams[\"team_name\"])" in source
