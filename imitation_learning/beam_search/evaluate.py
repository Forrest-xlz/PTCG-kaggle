"""Run the configured policy-only Beam-versus-Greedy matchup matrix."""
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
        f"beam={result.beam_deck} greedy={result.greedy_deck} "
        f"seat={result.beam_player} outcome={result.outcome} "
        f"seconds={result.duration_seconds:.2f} "
        f"searches={result.search_calls} nodes={result.expanded_nodes}"
        + (f" error={result.error}" if result.error else "")
    )


def main() -> None:
    settings = load_settings()
    games = schedule_games(settings)
    print(
        f"checkpoint={settings.checkpoint} decks={len(settings.decks)} "
        f"matchup_cells={len(settings.decks) ** 2} games={len(games):,} "
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
    names = tuple(deck.name for deck in settings.decks)
    report = aggregate_results(results, names)
    write_outputs(settings.output, results, report, names)
    overall = report.overall
    rate = overall["beam_win_rate"]
    rate_text = "N/A" if rate is None else f"{rate:.2%}"
    print(
        f"beam_overall_win_rate={rate_text} "
        f"wins={overall['beam_wins']:,} losses={overall['beam_losses']:,} "
        f"draws={overall['draws']:,} failures={overall['failures']:,} "
        f"output={settings.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
