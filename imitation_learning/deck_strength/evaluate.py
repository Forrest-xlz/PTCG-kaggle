"""Run the configured Greedy policy Deck-strength matchup matrix."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deck_strength.battle import GameResult
from deck_strength.config import load_settings, schedule_games
from deck_strength.reporting import aggregate_results, write_outputs
from deck_strength.runner import execute_games


def format_progress(completed: int, total: int, result: GameResult) -> str:
    return (
        f"[{completed:,}/{total:,}] game_id={result.game_id} "
        f"{result.deck_a} vs {result.deck_b} "
        f"a_player={result.deck_a_player} outcome_a={result.outcome_a} "
        f"seconds={result.duration_seconds:.2f} selections={result.selections}"
        + (f" error={result.error}" if result.error else "")
    )


def main() -> None:
    settings = load_settings()
    games = schedule_games(settings)
    pair_count = len(settings.decks) * (len(settings.decks) - 1) // 2
    print(
        f"checkpoint={settings.checkpoint} decks={len(settings.decks)} "
        f"pairs={pair_count} games={len(games):,} "
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

    print("\nDeck ranking:", flush=True)
    for rank, row in enumerate(report.ranking, start=1):
        rate = row["win_rate"]
        rate_text = "N/A" if rate is None else f"{rate:.2%}"
        print(
            f"{rank}. {row['deck']} win_rate={rate_text} "
            f"W-L-D={row['wins']}-{row['losses']}-{row['draws']} "
            f"failures={row['failures']}",
            flush=True,
        )
    print(f"output={settings.output}", flush=True)


if __name__ == "__main__":
    main()
