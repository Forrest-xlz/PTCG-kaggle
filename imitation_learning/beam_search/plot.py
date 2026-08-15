"""Focused reporting for a target Deck's Beam and Greedy conditions."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from beam_search.battle import GameResult


@dataclass(frozen=True, slots=True)
class AggregateReport:
    target_deck: str
    comparisons: tuple[dict[str, Any], ...]
    overall_beam: dict[str, Any]
    overall_greedy: dict[str, Any]
    overall_uplift: float | None


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
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "failures": failures,
        "win_rate": wins / decisive if decisive else None,
        "non_loss_rate": (wins + draws) / completed if completed else None,
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


def _prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def aggregate_results(
    results: Iterable[GameResult],
    target_deck: str,
    opponent_names: tuple[str, ...],
) -> AggregateReport:
    rows = list(results)
    comparisons: list[dict[str, Any]] = []
    for opponent_name in opponent_names:
        beam = _summary(
            [
                row
                for row in rows
                if row.condition == "beam"
                and row.opponent_deck == opponent_name
            ]
        )
        greedy = _summary(
            [
                row
                for row in rows
                if row.condition == "greedy"
                and row.opponent_deck == opponent_name
            ]
        )
        comparisons.append(
            {
                "target_deck": target_deck,
                "opponent_deck": opponent_name,
                **_prefixed("beam", beam),
                **_prefixed("greedy", greedy),
                "beam_uplift": _difference(
                    beam["win_rate"], greedy["win_rate"]
                ),
            }
        )
    overall_beam = _summary(
        [row for row in rows if row.condition == "beam"]
    )
    overall_greedy = _summary(
        [row for row in rows if row.condition == "greedy"]
    )
    return AggregateReport(
        target_deck=target_deck,
        comparisons=tuple(comparisons),
        overall_beam=overall_beam,
        overall_greedy=overall_greedy,
        overall_uplift=_difference(
            overall_beam["win_rate"], overall_greedy["win_rate"]
        ),
    )


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(path: Path, report: AggregateReport) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row["opponent_deck"] for row in report.comparisons]
    beam_values = [row["beam_win_rate"] for row in report.comparisons]
    greedy_values = [row["greedy_win_rate"] for row in report.comparisons]
    beam_plot = [float("nan") if value is None else value for value in beam_values]
    greedy_plot = [
        float("nan") if value is None else value for value in greedy_values
    ]
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(
        figsize=(max(7.0, len(labels) * 1.8), 5.5)
    )
    beam_bars = axis.bar(
        positions - width / 2, beam_plot, width, label="Beam Festival"
    )
    greedy_bars = axis.bar(
        positions + width / 2, greedy_plot, width, label="Greedy Festival"
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Festival Lead win rate (draws excluded)")
    axis.set_xlabel("Greedy opponent Deck")
    axis.set_title("Festival Lead: Beam Search vs Greedy Baseline")
    axis.set_xticks(positions, labels=labels, rotation=25, ha="right")
    axis.legend()
    for bars, prefix in ((beam_bars, "beam"), (greedy_bars, "greedy")):
        for bar, row in zip(bars, report.comparisons):
            rate = row[f"{prefix}_win_rate"]
            completed = row[f"{prefix}_completed_games"]
            label = f"N/A\nn={completed}" if rate is None else f"{rate:.1%}\nn={completed}"
            axis.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_outputs(
    output: Path,
    results: Iterable[GameResult],
    report: AggregateReport,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    comparison_rows = list(report.comparisons)
    _write_dict_rows(output / "games.csv", [asdict(row) for row in rows])
    _write_dict_rows(output / "opponent_summary.csv", comparison_rows)
    _write_dict_rows(output / "festival_comparison.csv", comparison_rows)
    payload = {
        "target_deck": report.target_deck,
        "overall_beam": report.overall_beam,
        "overall_greedy": report.overall_greedy,
        "overall_uplift": report.overall_uplift,
        "comparisons": comparison_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_comparison(output / "festival_win_rate_comparison.png", report)
