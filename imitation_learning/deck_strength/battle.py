"""Two-sided Greedy Top-1 CG battle execution."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from beam_search.model_agent import PolicyAgent, PolicyHistory
from deck_strength.config import GameSpec


@dataclass(frozen=True, slots=True)
class GameResult:
    game_id: int
    deck_a: str
    deck_b: str
    deck_a_player: int
    player0_deck: str
    player1_deck: str
    winner: int
    outcome_a: str
    outcome_b: str
    duration_seconds: float
    selections: int
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
    *,
    backend: Any | None = None,
) -> GameResult:
    engine = backend or CgBattleBackend()
    decks = {
        game.deck_a_player: game.deck_a,
        1 - game.deck_a_player: game.deck_b,
    }
    histories = {0: PolicyHistory(), 1: PolicyHistory()}
    started_at = time.perf_counter()
    started = False
    winner = -1
    selections = 0
    outcome_a = "failed"
    outcome_b = "failed"
    error = ""
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
            selected = policy.greedy(
                obs,
                decks[actor],
                histories[actor],
                record=True,
            )
            selections += 1
            raw_obs = engine.select(selected)

        if winner == 2:
            outcome_a = outcome_b = "draw"
        elif winner == game.deck_a_player:
            outcome_a, outcome_b = "win", "loss"
        else:
            outcome_a, outcome_b = "loss", "win"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if started:
            try:
                engine.finish()
            except Exception as exc:
                suffix = f"finish error: {type(exc).__name__}: {exc}"
                error = f"{error}; {suffix}" if error else suffix
                outcome_a = outcome_b = "failed"

    return GameResult(
        game_id=game.game_id,
        deck_a=game.deck_a.name,
        deck_b=game.deck_b.name,
        deck_a_player=game.deck_a_player,
        player0_deck=decks[0].name,
        player1_deck=decks[1].name,
        winner=winner,
        outcome_a=outcome_a,
        outcome_b=outcome_b,
        duration_seconds=time.perf_counter() - started_at,
        selections=selections,
        error=error,
    )
