"""Run the configured policy-only Beam-versus-Greedy matchup matrix."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beam_search.battle import CgBattleBackend, run_game
from beam_search.config import load_settings, schedule_games
from beam_search.model_agent import load_policy_agent
from beam_search.plot import aggregate_results, write_outputs
from beam_search.search import BeamSearcher, CgSearchBackend


def main() -> None:
    settings = load_settings()
    policy = load_policy_agent(settings)
    from cg.api import all_card_data

    basic_card_ids = frozenset(
        int(card.cardId) for card in all_card_data() if bool(card.basic)
    )
    searcher = BeamSearcher(policy, CgSearchBackend(), settings.search)
    battle_backend = CgBattleBackend()
    games = schedule_games(settings)
    print(
        f"checkpoint={settings.checkpoint} decks={len(settings.decks)} "
        f"matchup_cells={len(settings.decks) ** 2} games={len(games):,}",
        flush=True,
    )
    results = []
    for number, game in enumerate(games, start=1):
        result = run_game(
            game,
            policy,
            searcher,
            basic_card_ids,
            backend=battle_backend,
        )
        results.append(result)
        print(
            f"[{number:,}/{len(games):,}] "
            f"beam={result.beam_deck} greedy={result.greedy_deck} "
            f"seat={result.beam_player} outcome={result.outcome} "
            f"seconds={result.duration_seconds:.2f} "
            f"searches={result.search_calls} nodes={result.expanded_nodes}"
            + (f" error={result.error}" if result.error else ""),
            flush=True,
        )
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
