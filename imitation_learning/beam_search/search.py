"""Turn-level policy-only beam search over CG Search States."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from beam_search.config import DeckSpec, SearchSettings
from beam_search.determinization import SearchInputs
from beam_search.model_agent import PolicyAgent, PolicyHistory, PolicyOutput


class SearchBackend(Protocol):
    def begin(self, obs: Any, inputs: SearchInputs) -> Any: ...
    def step(self, search_id: int, action: list[int]) -> Any: ...
    def release(self, search_id: int) -> None: ...
    def end(self) -> None: ...


class CgSearchBackend:
    def begin(self, obs: Any, inputs: SearchInputs) -> Any:
        from cg.api import search_begin

        return search_begin(
            obs,
            your_deck=list(inputs.your_deck),
            your_prize=list(inputs.your_prize),
            opponent_deck=list(inputs.opponent_deck),
            opponent_prize=list(inputs.opponent_prize),
            opponent_hand=list(inputs.opponent_hand),
            opponent_active=list(inputs.opponent_active),
        )

    def step(self, search_id: int, action: list[int]) -> Any:
        from cg.api import search_step

        return search_step(search_id, action)

    def release(self, search_id: int) -> None:
        from cg.api import search_release

        search_release(search_id)

    def end(self) -> None:
        from cg.api import search_end

        search_end()


@dataclass(frozen=True, slots=True)
class SearchStats:
    search_calls: int = 0
    expanded_nodes: int = 0
    max_depth: int = 0
    truncated_trajectories: int = 0
    branch_errors: int = 0
    greedy_fallback: bool = False
    selected_scored_steps: int = 0
    selected_opponent_steps: int = 0


@dataclass(frozen=True, slots=True)
class SearchDecision:
    action: list[int]
    score: float | None
    stats: SearchStats


@dataclass(slots=True)
class _Trajectory:
    state: Any
    first_action: list[int] | None
    log_sum: float
    scored_steps: int
    opponent_steps: int
    histories: dict[int, PolicyHistory]
    depth: int
    truncated: bool = False


def trajectory_score(log_sum: float, steps: int, alpha: float) -> float:
    if steps < 1:
        raise ValueError("trajectory steps must be >= 1")
    return float(log_sum) / (float(steps) ** float(alpha))


def _ranked_indices(output: PolicyOutput, count: int) -> list[int]:
    return sorted(
        range(len(output.actions)),
        key=lambda index: (-float(output.probabilities[index].item()), index),
    )[:count]


class BeamSearcher:
    def __init__(
        self,
        policy: PolicyAgent,
        backend: SearchBackend,
        settings: SearchSettings,
    ) -> None:
        self.policy = policy
        self.backend = backend
        self.settings = settings

    def _partial_score(self, trajectory: _Trajectory) -> float:
        return trajectory_score(
            trajectory.log_sum,
            trajectory.scored_steps,
            self.settings.alpha,
        )

    def choose(
        self,
        obs: Any,
        *,
        root_player: int,
        root_deck: DeckSpec,
        opponent_deck: DeckSpec,
        root_history: PolicyHistory,
        opponent_history: PolicyHistory,
        search_inputs: SearchInputs,
    ) -> SearchDecision:
        fallback_output = self.policy.evaluate(obs, root_deck, root_history)
        fallback = fallback_output.actions[
            int(fallback_output.probabilities.argmax().item())
        ]
        if int(obs.current.turn) == 0:
            return SearchDecision(
                fallback,
                None,
                SearchStats(greedy_fallback=True),
            )

        stats = SearchStats(search_calls=1)
        owned_ids: set[int] = set()

        def release(search_id: int) -> None:
            if search_id in owned_ids:
                self.backend.release(search_id)
                owned_ids.remove(search_id)

        try:
            root_state = self.backend.begin(obs, search_inputs)
            owned_ids.add(int(root_state.searchId))
            histories = {
                root_player: root_history.clone(),
                1 - root_player: opponent_history.clone(),
            }
            decks = {
                root_player: root_deck,
                1 - root_player: opponent_deck,
            }
            root_turn = int(obs.current.turn)
            live = [
                _Trajectory(
                    state=root_state,
                    first_action=None,
                    log_sum=0.0,
                    scored_steps=0,
                    opponent_steps=0,
                    histories=histories,
                    depth=0,
                )
            ]
            completed: list[_Trajectory] = []
            while live:
                next_live: list[_Trajectory] = []
                for trajectory in live:
                    state = trajectory.state
                    observation = state.observation
                    actor = int(observation.current.yourIndex)
                    output = self.policy.evaluate(
                        observation,
                        decks[actor],
                        trajectory.histories[actor],
                    )
                    expansion_count = (
                        self.settings.expansion_top_k
                        if actor == root_player
                        else 1
                    )
                    for action_index in _ranked_indices(
                        output, expansion_count
                    ):
                        action = output.actions[action_index]
                        try:
                            child_state = self.backend.step(
                                int(state.searchId), action
                            )
                        except Exception:
                            stats = replace(
                                stats, branch_errors=stats.branch_errors + 1
                            )
                            continue
                        owned_ids.add(int(child_state.searchId))
                        child_histories = {
                            player: history.clone()
                            for player, history in trajectory.histories.items()
                        }
                        self.policy.record_action(
                            observation,
                            action,
                            child_histories[actor],
                            output.encoded_options,
                        )
                        child = _Trajectory(
                            state=child_state,
                            first_action=(
                                list(action)
                                if trajectory.first_action is None
                                else trajectory.first_action
                            ),
                            log_sum=(
                                trajectory.log_sum
                                + float(
                                    output.log_probabilities[
                                        action_index
                                    ].item()
                                )
                                if actor == root_player
                                else trajectory.log_sum
                            ),
                            scored_steps=(
                                trajectory.scored_steps
                                + int(actor == root_player)
                            ),
                            opponent_steps=(
                                trajectory.opponent_steps
                                + int(actor != root_player)
                            ),
                            histories=child_histories,
                            depth=trajectory.depth + 1,
                        )
                        stats = replace(
                            stats,
                            expanded_nodes=stats.expanded_nodes + 1,
                            max_depth=max(stats.max_depth, child.depth),
                        )
                        child_obs = child_state.observation
                        terminal = int(child_obs.current.result) >= 0
                        turn_finished = int(child_obs.current.turn) != root_turn
                        depth_finished = child.depth >= self.settings.max_depth
                        if terminal or turn_finished or depth_finished:
                            child.truncated = depth_finished and not (
                                terminal or turn_finished
                            )
                            completed.append(child)
                            if child.truncated:
                                stats = replace(
                                    stats,
                                    truncated_trajectories=(
                                        stats.truncated_trajectories + 1
                                    ),
                                )
                            release(int(child_state.searchId))
                        else:
                            next_live.append(child)
                    release(int(state.searchId))
                next_live.sort(key=self._partial_score, reverse=True)
                for pruned in next_live[self.settings.beam_width :]:
                    release(int(pruned.state.searchId))
                live = next_live[: self.settings.beam_width]

            eligible = [
                trajectory
                for trajectory in completed
                if trajectory.first_action is not None
                and trajectory.scored_steps >= 1
            ]
            if not eligible:
                return SearchDecision(
                    fallback,
                    None,
                    replace(stats, greedy_fallback=True),
                )
            best = max(eligible, key=self._partial_score)
            return SearchDecision(
                list(best.first_action),
                self._partial_score(best),
                replace(
                    stats,
                    selected_scored_steps=best.scored_steps,
                    selected_opponent_steps=best.opponent_steps,
                ),
            )
        except Exception:
            return SearchDecision(
                fallback,
                None,
                replace(
                    stats,
                    branch_errors=stats.branch_errors + 1,
                    greedy_fallback=True,
                ),
            )
        finally:
            for search_id in list(owned_ids):
                try:
                    release(search_id)
                except Exception:
                    owned_ids.discard(search_id)
            try:
                self.backend.end()
            except Exception:
                pass
