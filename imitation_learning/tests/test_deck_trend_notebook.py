from __future__ import annotations

import ast
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "deck_trends.ipynb"


def test_project_root_discovery_uses_the_reorganized_analysis_package(
    monkeypatch,
) -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    setup_source = next(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    tree = ast.parse(setup_source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "find_project_root"
    )
    namespace = {"Path": Path}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(NOTEBOOK_PATH),
            "exec",
        ),
        namespace,
    )

    monkeypatch.chdir(PROJECT_ROOT.parent)

    assert namespace["find_project_root"]() == PROJECT_ROOT


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
            "plot_archetype_sankey",
            "build_matchups",
            "daily_share_visibility",
            "build_pooled_archetype_shares",
            "global_visible_archetypes",
            "matchup_matrix_long.csv",
        )
    )
    assert "groupby('archetype')['share_percent'].max()" not in source
    assert "go.Sankey" not in source
