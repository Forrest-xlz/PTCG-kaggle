"""Single-process CG battle execution for Beam-versus-Greedy games."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from beam_search.config import GameSpec
from beam_search.determinization import SearchInputs, build_search_inputs
from beam_search.model_agent import PolicyAgent, PolicyHistory
from beam_search.search import BeamSearcher


@dataclass(frozen=True, slots=True)
class GameResult:
    game_id: int
    condition: str
    target_deck: str
    opponent_deck: str
    target_player: int
    player0_deck: str
    player1_deck: str
    outcome: str
    winner: int
    duration_seconds: float
    selections: int
    search_calls: int
    expanded_nodes: int
    max_depth: int
    truncated_trajectories: int
    branch_errors: int
    greedy_fallbacks: int
    scored_steps: int
    opponent_steps: int
    warnings: str
    error: str


class CgBattleBackend:
    def start(self, deck0: list[int], deck1: list[int]):
        from cg.game import battle_start

        return battle_start(deck0, deck1)

    def select(self, action: list[int]):
        from cg.game import battle_select

        return battle_select(action)

    def finish(self) -> None:
        from cg.game import battle_finish

        battle_finish()

    def observation(self, value: Any):
        from cg.api import to_observation_class

        return to_observation_class(value)


def run_game(
    game: GameSpec,
    policy: PolicyAgent,
    searcher: BeamSearcher,
    basic_card_ids: frozenset[int],
    *,
    backend: Any | None = None,
    determinize: Callable[..., SearchInputs] = build_search_inputs,
) -> GameResult:
    engine = backend or CgBattleBackend()
    decks = {
        game.target_player: game.target_deck,
        1 - game.target_player: game.opponent_deck,
    }
    histories = {0: PolicyHistory(), 1: PolicyHistory()}
    counters = {
        "selections": 0,
        "search_calls": 0,
        "expanded_nodes": 0,
        "max_depth": 0,
        "truncated_trajectories": 0,
        "branch_errors": 0,
        "greedy_fallbacks": 0,
        "scored_steps": 0,
        "opponent_steps": 0,
    }
    warning_messages: list[str] = []
    started = False
    winner = -1
    outcome = "failed"
    error = ""
    started_at = time.perf_counter()
    try:
        raw_obs, start_data = engine.start(
            list(decks[0].cards), list(decks[1].cards)
        )
        if int(start_data.errorPlayer) >= 0:
            raise ValueError(
                "battle_start rejected deck for player "
                f"{start_data.errorPlayer}, errorType={start_data.errorType}"
            )
        if raw_obs is None:
            raise RuntimeError("battle_start returned no observation")
        started = True
        while True:
            obs = engine.observation(raw_obs)
            if int(obs.current.result) >= 0:
                winner = int(obs.current.result)
                break
            actor = int(obs.current.yourIndex)
            use_beam = (
                game.condition == "beam"
                and actor == game.target_player
                and int(obs.current.turn) > 0
            )
            if use_beam:
                try:
                    inputs = determinize(
                        obs,
                        decks[actor],
                        decks[1 - actor],
                        random.Random(
                            (int(game.seed) << 32) + counters["search_calls"]
                        ),
                        basic_card_ids=basic_card_ids,
                    )
                    warning_messages.extend(inputs.warnings)
                    decision = searcher.choose(
                        obs,
                        root_player=actor,
                        root_deck=decks[actor],
                        opponent_deck=decks[1 - actor],
                        root_history=histories[actor],
                        opponent_history=histories[1 - actor],
                        search_inputs=inputs,
                    )
                    selected = decision.action
                    stats = decision.stats
                    counters["search_calls"] += stats.search_calls
                    counters["expanded_nodes"] += stats.expanded_nodes
                    counters["max_depth"] = max(
                        counters["max_depth"], stats.max_depth
                    )
                    counters["truncated_trajectories"] += (
                        stats.truncated_trajectories
                    )
                    counters["branch_errors"] += stats.branch_errors
                    counters["greedy_fallbacks"] += int(stats.greedy_fallback)
                    counters["scored_steps"] += stats.selected_scored_steps
                    counters["opponent_steps"] += stats.selected_opponent_steps
                except Exception as exc:
                    warning_messages.append(
                        f"determinization/search fallback: {type(exc).__name__}: {exc}"
                    )
                    counters["greedy_fallbacks"] += 1
                    selected = policy.greedy(
                        obs, decks[actor], histories[actor], record=False
                    )
                policy.record_action(obs, selected, histories[actor])
            else:
                selected = policy.greedy(
                    obs, decks[actor], histories[actor], record=True
                )
            counters["selections"] += 1
            raw_obs = engine.select(selected)
        if winner == 2:
            outcome = "draw"
        elif winner == game.target_player:
            outcome = "win"
        else:
            outcome = "loss"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if started:
            try:
                engine.finish()
            except Exception as exc:
                suffix = f"finish error: {type(exc).__name__}: {exc}"
                error = f"{error}; {suffix}" if error else suffix
                outcome = "failed"
    return GameResult(
        game_id=game.game_id,
        condition=game.condition,
        target_deck=game.target_deck.name,
        opponent_deck=game.opponent_deck.name,
        target_player=game.target_player,
        player0_deck=decks[0].name,
        player1_deck=decks[1].name,
        outcome=outcome,
        winner=winner,
        duration_seconds=time.perf_counter() - started_at,
        selections=counters["selections"],
        search_calls=counters["search_calls"],
        expanded_nodes=counters["expanded_nodes"],
        max_depth=counters["max_depth"],
        truncated_trajectories=counters["truncated_trajectories"],
        branch_errors=counters["branch_errors"],
        greedy_fallbacks=counters["greedy_fallbacks"],
        scored_steps=counters["scored_steps"],
        opponent_steps=counters["opponent_steps"],
        warnings=" | ".join(warning_messages),
        error=error,
    )
