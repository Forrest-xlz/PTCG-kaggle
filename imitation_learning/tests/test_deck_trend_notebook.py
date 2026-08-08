from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "eda" / "deck_trend.ipynb"


def test_deck_trend_notebook_is_valid_and_complete() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    markdown = "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    source = "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    headings = [
        "## Goal",
        "## Setup and Configuration",
        "## Load and Validate Extracted Data",
        "## Select Date Snapshots",
        "## Classification Coverage",
        "## Archetype Usage and Win Rate",
        "## Team Modal Archetypes and Retention",
        "## Sankey Flow",
        "## Matchup Matrix",
        "## Exported Tables",
        "## Takeaways",
    ]
    assert all(heading in markdown for heading in headings)
    assert all(
        symbol in source
        for symbol in (
            "SNAPSHOT_INTERVAL_DAYS",
            "add_trend_archetypes",
            "build_daily_metrics",
            "build_team_flows",
            "go.Sankey",
            "build_matchups",
            "matchup_matrix_long.csv",
        )
    )
