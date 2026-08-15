"""CSV, JSON, and matrix-plot reporting for Beam-versus-Greedy games."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from beam_search.battle import GameResult


@dataclass(frozen=True, slots=True)
class AggregateReport:
    matchups: tuple[dict[str, Any], ...]
    matrix: dict[tuple[str, str], float | None]
    completed: dict[tuple[str, str], int]
    overall: dict[str, Any]
    by_beam_deck: dict[str, dict[str, Any]]
    by_greedy_deck: dict[str, dict[str, Any]]


def _summary(rows: list[GameResult]) -> dict[str, Any]:
    wins = sum(row.outcome == "win" for row in rows)
    losses = sum(row.outcome == "loss" for row in rows)
    draws = sum(row.outcome == "draw" for row in rows)
    failures = sum(row.outcome == "failed" for row in rows)
    completed = wins + losses + draws
    decisive = wins + losses
    durations = [
        row.duration_seconds for row in rows if row.outcome != "failed"
    ]
    return {
        "attempted_games": len(rows),
        "completed_games": completed,
        "beam_wins": wins,
        "beam_losses": losses,
        "draws": draws,
        "failures": failures,
        "beam_win_rate": wins / decisive if decisive else None,
        "beam_non_loss_rate": (wins + draws) / completed if completed else None,
        "mean_duration_seconds": (
            sum(durations) / len(durations) if durations else None
        ),
        "search_calls": sum(row.search_calls for row in rows),
        "expanded_nodes": sum(row.expanded_nodes for row in rows),
        "truncated_trajectories": sum(
            row.truncated_trajectories for row in rows
        ),
        "branch_errors": sum(row.branch_errors for row in rows),
        "greedy_fallbacks": sum(row.greedy_fallbacks for row in rows),
    }


def aggregate_results(
    results: Iterable[GameResult],
    deck_names: tuple[str, ...],
) -> AggregateReport:
    rows = list(results)
    matrix: dict[tuple[str, str], float | None] = {}
    completed: dict[tuple[str, str], int] = {}
    matchups: list[dict[str, Any]] = []
    for beam_name in deck_names:
        for greedy_name in deck_names:
            cell_rows = [
                row
                for row in rows
                if row.beam_deck == beam_name
                and row.greedy_deck == greedy_name
            ]
            values = _summary(cell_rows)
            matchups.append(
                {
                    "beam_deck": beam_name,
                    "greedy_deck": greedy_name,
                    **values,
                }
            )
            key = (beam_name, greedy_name)
            matrix[key] = values["beam_win_rate"]
            completed[key] = values["completed_games"]
    return AggregateReport(
        matchups=tuple(matchups),
        matrix=matrix,
        completed=completed,
        overall=_summary(rows),
        by_beam_deck={
            name: _summary([row for row in rows if row.beam_deck == name])
            for name in deck_names
        },
        by_greedy_deck={
            name: _summary([row for row in rows if row.greedy_deck == name])
            for name in deck_names
        },
    )


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_matrix(
    path: Path,
    report: AggregateReport,
    deck_names: tuple[str, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = len(deck_names)
    values = np.full((size, size), np.nan, dtype=np.float64)
    for row, beam_name in enumerate(deck_names):
        for column, greedy_name in enumerate(deck_names):
            value = report.matrix[(beam_name, greedy_name)]
            if value is not None:
                values[row, column] = value
    figure, axis = plt.subplots(
        figsize=(max(6.0, size * 1.7), max(5.0, size * 1.35))
    )
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    axis.set_xticks(range(size), labels=deck_names, rotation=30, ha="right")
    axis.set_yticks(range(size), labels=deck_names)
    axis.set_xlabel("Greedy deck")
    axis.set_ylabel("Beam deck")
    axis.set_title("Beam Search Win Rate vs Greedy")
    for row, beam_name in enumerate(deck_names):
        for column, greedy_name in enumerate(deck_names):
            key = (beam_name, greedy_name)
            value = report.matrix[key]
            text = (
                "N/A\n"
                if value is None
                else f"{value:.1%}\n"
            ) + f"n={report.completed[key]}"
            axis.text(column, row, text, ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Beam win rate (draws excluded)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_outputs(
    output: Path,
    results: Iterable[GameResult],
    report: AggregateReport,
    deck_names: tuple[str, ...],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    _write_dict_rows(output / "games.csv", [asdict(row) for row in rows])
    _write_dict_rows(
        output / "matchup_summary.csv", list(report.matchups)
    )
    with (output / "beam_win_rate_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("beam_deck", *deck_names))
        for beam_name in deck_names:
            writer.writerow(
                (
                    beam_name,
                    *(
                        ""
                        if report.matrix[(beam_name, greedy_name)] is None
                        else report.matrix[(beam_name, greedy_name)]
                        for greedy_name in deck_names
                    ),
                )
            )
    payload = {
        "overall": report.overall,
        "by_beam_deck": report.by_beam_deck,
        "by_greedy_deck": report.by_greedy_deck,
        "matchups": list(report.matchups),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_matrix(
        output / "beam_win_rate_matrix.png", report, deck_names
    )

