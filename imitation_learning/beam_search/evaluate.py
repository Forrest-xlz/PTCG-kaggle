"""Compare Beam and Greedy Festival Lead against configured opponents."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beam_search.battle import GameResult
from beam_search.config import load_settings, schedule_games
from beam_search.plot import aggregate_results, write_outputs
from beam_search.runner import execute_games


def format_progress(completed: int, total: int, result: GameResult) -> str:
    return (
        f"[{completed:,}/{total:,}] game_id={result.game_id} "
        f"condition={result.condition} target={result.target_deck} "
        f"opponent={result.opponent_deck} seat={result.target_player} "
        f"outcome={result.outcome} "
        f"seconds={result.duration_seconds:.2f} "
        f"searches={result.search_calls} nodes={result.expanded_nodes}"
        + (f" error={result.error}" if result.error else "")
    )


def main() -> None:
    settings = load_settings()
    games = schedule_games(settings)
    print(
        f"checkpoint={settings.checkpoint} target={settings.target_deck} "
        f"opponents={len(settings.decks) - 1} conditions=2 "
        f"games={len(games):,} "
        f"workers={settings.runtime.workers} device={settings.device}",
        flush=True,
    )
    if settings.runtime.workers > 1 and settings.device.lower().startswith("cuda"):
        print(
            "warning: each worker loads an independent model on the same CUDA "
            "device; GPU memory use grows with worker count",
            flush=True,
        )

    completed = 0

    def report_progress(result: GameResult) -> None:
        nonlocal completed
        completed += 1
        print(format_progress(completed, len(games), result), flush=True)

    results = execute_games(settings, games, on_result=report_progress)
    opponents = tuple(
        deck.name
        for deck in settings.decks
        if deck.name != settings.target_deck
    )
    report = aggregate_results(results, settings.target_deck, opponents)
    write_outputs(settings.output, results, report)

    def percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    search_calls = report.overall_beam["search_calls"]
    fallback_rate = (
        report.overall_beam["greedy_fallbacks"] / search_calls
        if search_calls
        else None
    )
    print(
        f"beam_win_rate={percent(report.overall_beam['win_rate'])} "
        f"greedy_win_rate={percent(report.overall_greedy['win_rate'])} "
        f"uplift={percent(report.overall_uplift)} "
        f"fallback_rate={percent(fallback_rate)} "
        f"output={settings.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
