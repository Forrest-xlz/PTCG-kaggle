"""Aggregate and render directed Greedy Deck-strength results."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from deck_strength.battle import GameResult


@dataclass(frozen=True, slots=True)
class AggregateReport:
    matchups: tuple[dict[str, Any], ...]
    matrix: dict[tuple[str, str], float | None]
    completed: dict[tuple[str, str], int]
    ranking: tuple[dict[str, Any], ...]
    overall: dict[str, Any]


def _perspective_summary(
    rows: list[tuple[GameResult, str]],
    deck_name: str,
) -> dict[str, Any]:
    outcomes = [outcome for _, outcome in rows]
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    draws = outcomes.count("draw")
    failures = outcomes.count("failed")
    completed = wins + losses + draws
    decisive = wins + losses
    durations = [
        row.duration_seconds
        for row, outcome in rows
        if outcome != "failed"
    ]
    return {
        "deck": deck_name,
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
    }


def _overall_summary(rows: list[GameResult]) -> dict[str, Any]:
    failures = sum(row.outcome_a == "failed" for row in rows)
    draws = sum(row.outcome_a == "draw" for row in rows)
    completed = len(rows) - failures
    durations = [
        row.duration_seconds for row in rows if row.outcome_a != "failed"
    ]
    return {
        "attempted_games": len(rows),
        "completed_games": completed,
        "draws": draws,
        "failures": failures,
        "mean_duration_seconds": (
            sum(durations) / len(durations) if durations else None
        ),
        "selections": sum(row.selections for row in rows),
    }


def aggregate_results(
    results: Iterable[GameResult],
    deck_names: tuple[str, ...],
) -> AggregateReport:
    rows = list(results)
    matrix = {
        (row_name, column_name): None
        for row_name in deck_names
        for column_name in deck_names
    }
    completed = {
        (row_name, column_name): 0
        for row_name in deck_names
        for column_name in deck_names
    }
    matchups: list[dict[str, Any]] = []

    for left_index, deck_a in enumerate(deck_names):
        for deck_b in deck_names[left_index + 1 :]:
            pair_rows = [
                row
                for row in rows
                if row.deck_a == deck_a and row.deck_b == deck_b
            ]
            a_wins = sum(row.outcome_a == "win" for row in pair_rows)
            b_wins = sum(row.outcome_b == "win" for row in pair_rows)
            draws = sum(row.outcome_a == "draw" for row in pair_rows)
            failures = sum(row.outcome_a == "failed" for row in pair_rows)
            pair_completed = a_wins + b_wins + draws
            decisive = a_wins + b_wins
            durations = [
                row.duration_seconds
                for row in pair_rows
                if row.outcome_a != "failed"
            ]
            a_rate = a_wins / decisive if decisive else None
            b_rate = b_wins / decisive if decisive else None
            matchups.append(
                {
                    "deck_a": deck_a,
                    "deck_b": deck_b,
                    "attempted_games": len(pair_rows),
                    "completed_games": pair_completed,
                    "deck_a_wins": a_wins,
                    "deck_b_wins": b_wins,
                    "draws": draws,
                    "failures": failures,
                    "deck_a_win_rate": a_rate,
                    "deck_b_win_rate": b_rate,
                    "deck_a_non_loss_rate": (
                        (a_wins + draws) / pair_completed
                        if pair_completed
                        else None
                    ),
                    "deck_b_non_loss_rate": (
                        (b_wins + draws) / pair_completed
                        if pair_completed
                        else None
                    ),
                    "mean_duration_seconds": (
                        sum(durations) / len(durations) if durations else None
                    ),
                }
            )
            matrix[(deck_a, deck_b)] = a_rate
            matrix[(deck_b, deck_a)] = b_rate
            completed[(deck_a, deck_b)] = pair_completed
            completed[(deck_b, deck_a)] = pair_completed

    ranking: list[dict[str, Any]] = []
    for name in deck_names:
        perspective_rows = []
        for row in rows:
            if row.deck_a == name:
                perspective_rows.append((row, row.outcome_a))
            elif row.deck_b == name:
                perspective_rows.append((row, row.outcome_b))
        ranking.append(_perspective_summary(perspective_rows, name))
    ranking.sort(
        key=lambda row: (
            row["win_rate"] is None,
            -(row["win_rate"] or 0.0),
            -row["wins"],
            row["deck"],
        )
    )

    return AggregateReport(
        matchups=tuple(matchups),
        matrix=matrix,
        completed=completed,
        ranking=tuple(ranking),
        overall=_overall_summary(rows),
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
    for row, deck_name in enumerate(deck_names):
        for column, opponent_name in enumerate(deck_names):
            value = report.matrix[(deck_name, opponent_name)]
            if value is not None:
                values[row, column] = value

    color_map = plt.get_cmap("RdYlGn").copy()
    color_map.set_bad("#dddddd")
    figure, axis = plt.subplots(
        figsize=(max(6.0, size * 1.7), max(5.0, size * 1.35))
    )
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap=color_map)
    axis.set_xticks(range(size), labels=deck_names, rotation=30, ha="right")
    axis.set_yticks(range(size), labels=deck_names)
    axis.set_xlabel("Opponent Deck")
    axis.set_ylabel("Deck")
    axis.set_title("Greedy Policy Deck Win Rate")
    for row, deck_name in enumerate(deck_names):
        for column, opponent_name in enumerate(deck_names):
            if deck_name == opponent_name:
                text = "—"
            else:
                value = report.matrix[(deck_name, opponent_name)]
                text = (
                    "N/A\n" if value is None else f"{value:.1%}\n"
                ) + f"n={report.completed[(deck_name, opponent_name)]}"
            axis.text(column, row, text, ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Win rate (draws excluded)")
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
    _write_dict_rows(output / "matchup_summary.csv", list(report.matchups))
    _write_dict_rows(output / "deck_ranking.csv", list(report.ranking))
    with (output / "win_rate_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("deck", *deck_names))
        for deck_name in deck_names:
            writer.writerow(
                (
                    deck_name,
                    *(
                        ""
                        if report.matrix[(deck_name, opponent)] is None
                        else report.matrix[(deck_name, opponent)]
                        for opponent in deck_names
                    ),
                )
            )
    payload = {
        "overall": report.overall,
        "matchups": list(report.matchups),
        "ranking": list(report.ranking),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_matrix(output / "win_rate_matrix.png", report, deck_names)
