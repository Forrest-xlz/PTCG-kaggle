from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beam_search.config import DeckSpec, SearchSettings
from beam_search.determinization import SearchInputs
from beam_search.model_agent import PolicyHistory, PolicyOutput
from beam_search.search import BeamSearcher, trajectory_score


def obs(name: str, *, actor: int, turn: int = 1, result: int = -1):
    return SimpleNamespace(
        name=name,
        current=SimpleNamespace(yourIndex=actor, turn=turn, result=result),
        select=SimpleNamespace(),
    )


@dataclass(frozen=True)
class FakeState:
    searchId: int
    observation: object


class FakeBackend:
    def __init__(self, root, transitions):
        self.root = root
        self.transitions = transitions
        self.released: list[int] = []
        self.ended = 0

    def begin(self, observation, inputs):
        return self.root

    def step(self, search_id: int, action: list[int]):
        value = self.transitions[(search_id, tuple(action))]
        if isinstance(value, Exception):
            raise value
        return value

    def release(self, search_id: int) -> None:
        self.released.append(search_id)

    def end(self) -> None:
        self.ended += 1


class FakePolicy:
    def __init__(self, outputs):
        self.outputs = outputs
        self.evaluated: list[tuple[str, str]] = []

    def evaluate(self, observation, deck, history):
        self.evaluated.append((observation.name, deck.name))
        actions, probabilities = self.outputs[observation.name]
        probs = torch.tensor(probabilities, dtype=torch.float32)
        return PolicyOutput(
            actions=tuple([*action] for action in actions),
            probabilities=probs,
            log_probabilities=probs.log(),
        )

    def record_action(self, observation, selected, history, encoded_options=None):
        history.actions.append(
            SimpleNamespace(select_type=len(selected), select_context=0)
        )

    def greedy(self, observation, deck, history, *, record=False):
        output = self.evaluate(observation, deck, history)
        return output.actions[int(output.probabilities.argmax())]


INPUTS = SearchInputs((), (), (), (), (), (), ())
DECK_A = DeckSpec("a", tuple(range(60)))
DECK_B = DeckSpec("b", tuple(range(60)))


def settings(**overrides) -> SearchSettings:
    values = dict(beam_width=2, expansion_top_k=2, alpha=1.0, max_depth=8)
    values.update(overrides)
    return SearchSettings(**values)


def test_score_is_length_normalized() -> None:
    assert trajectory_score(-6.0, 3, 1.0) == pytest.approx(-2.0)
    assert trajectory_score(-6.0, 3, 0.0) == pytest.approx(-6.0)
    with pytest.raises(ValueError, match="steps"):
        trajectory_score(-1.0, 0, 1.0)


def test_beam_search_scores_only_root_player_actions_and_returns_first() -> None:
    root = FakeState(0, obs("root", actor=0))
    opponent = FakeState(1, obs("opponent", actor=1))
    alternate = FakeState(2, obs("alternate", actor=0))
    after_opponent = FakeState(3, obs("after_opponent", actor=0))
    end_a = FakeState(4, obs("end_a", actor=1, turn=2))
    end_b = FakeState(5, obs("end_b", actor=1, turn=2))
    backend = FakeBackend(
        root,
        {
            (0, (0,)): opponent,
            (0, (1,)): alternate,
            (1, (0,)): after_opponent,
            (3, (0,)): end_a,
            (2, (0,)): end_b,
        },
    )
    policy = FakePolicy(
        {
            "root": (([0], [1]), (0.6, 0.4)),
            "opponent": (([0],), (0.01,)),
            "after_opponent": (([0],), (0.9,)),
            "alternate": (([0],), (0.99,)),
        }
    )
    searcher = BeamSearcher(policy, backend, settings())

    decision = searcher.choose(
        root.observation,
        root_player=0,
        root_deck=DECK_A,
        opponent_deck=DECK_B,
        root_history=PolicyHistory(),
        opponent_history=PolicyHistory(),
        search_inputs=INPUTS,
    )

    assert decision.action == [0]
    assert decision.stats.selected_scored_steps == 2
    assert decision.stats.selected_opponent_steps == 1
    assert ("opponent", "b") in policy.evaluated
    assert backend.ended == 1
    assert sorted(backend.released) == [0, 1, 2, 3, 4, 5]


def test_max_depth_completes_trajectory_and_counts_truncation() -> None:
    root = FakeState(0, obs("root", actor=0))
    child = FakeState(1, obs("child", actor=0))
    backend = FakeBackend(root, {(0, (0,)): child})
    policy = FakePolicy({"root": (([0],), (1.0,))})
    searcher = BeamSearcher(policy, backend, settings(max_depth=1))

    decision = searcher.choose(
        root.observation,
        root_player=0,
        root_deck=DECK_A,
        opponent_deck=DECK_B,
        root_history=PolicyHistory(),
        opponent_history=PolicyHistory(),
        search_inputs=INPUTS,
    )

    assert decision.action == [0]
    assert decision.stats.truncated_trajectories == 1
    assert decision.stats.max_depth == 1


def test_all_failed_branches_fall_back_to_root_greedy() -> None:
    root = FakeState(0, obs("root", actor=0))
    backend = FakeBackend(
        root,
        {
            (0, (0,)): RuntimeError("bad branch"),
            (0, (1,)): RuntimeError("bad branch"),
        },
    )
    policy = FakePolicy({"root": (([0], [1]), (0.2, 0.8))})
    searcher = BeamSearcher(policy, backend, settings())

    decision = searcher.choose(
        root.observation,
        root_player=0,
        root_deck=DECK_A,
        opponent_deck=DECK_B,
        root_history=PolicyHistory(),
        opponent_history=PolicyHistory(),
        search_inputs=INPUTS,
    )

    assert decision.action == [1]
    assert decision.stats.greedy_fallback
    assert decision.stats.branch_errors == 2
    assert backend.ended == 1
