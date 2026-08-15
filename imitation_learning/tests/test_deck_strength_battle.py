from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deck_strength.battle import run_game
from deck_strength.config import DeckSpec, GameSpec


def _observation(*, actor: int, result: int = -1):
    return SimpleNamespace(
        current=SimpleNamespace(yourIndex=actor, result=result),
        select=None if result >= 0 else SimpleNamespace(),
    )


class Backend:
    def __init__(self, observations) -> None:
        self.observations = list(observations)
        self.selected: list[list[int]] = []
        self.finished = 0

    def start(self, deck0, deck1):
        return self.observations.pop(0), SimpleNamespace(errorPlayer=-1, errorType=0)

    def select(self, action):
        self.selected.append(action)
        return self.observations.pop(0)

    def finish(self):
        self.finished += 1

    def observation(self, value):
        return value


class Policy:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int]] = []

    def greedy(self, observation, deck, history, *, record=False):
        self.calls.append((observation.current.yourIndex, deck.name, id(history)))
        assert record is True
        return [0]


def test_run_game_uses_each_seats_deck_and_independent_histories() -> None:
    deck_a = DeckSpec("a", tuple(range(60)))
    deck_b = DeckSpec("b", tuple(reversed(range(60))))
    game = GameSpec(4, deck_a, deck_b, deck_a_player=1, seed=20)
    backend = Backend(
        [
            _observation(actor=0),
            _observation(actor=1),
            _observation(actor=0, result=1),
        ]
    )
    policy = Policy()

    result = run_game(game, policy, backend=backend)

    assert [(actor, deck) for actor, deck, _ in policy.calls] == [(0, "b"), (1, "a")]
    assert policy.calls[0][2] != policy.calls[1][2]
    assert result.outcome_a == "win"
    assert result.outcome_b == "loss"
    assert result.winner == 1
    assert result.selections == 2
    assert result.player0_deck == "b"
    assert result.player1_deck == "a"
    assert backend.finished == 1


def test_run_game_records_start_failure_without_finish() -> None:
    deck = DeckSpec("a", tuple(range(60)))
    game = GameSpec(0, deck, deck, deck_a_player=0, seed=1)

    class FailingBackend:
        finished = 0

        def start(self, deck0, deck1):
            raise RuntimeError("engine unavailable")

        def finish(self):
            self.finished += 1

    backend = FailingBackend()
    result = run_game(game, Policy(), backend=backend)

    assert result.outcome_a == "failed"
    assert result.outcome_b == "failed"
    assert result.error == "RuntimeError: engine unavailable"
    assert backend.finished == 0
