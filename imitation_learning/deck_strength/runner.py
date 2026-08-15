"""Persistent sequential or spawn-based workers for Deck strength games."""
from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Executor, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

import torch

from beam_search.model_agent import load_policy_agent
from deck_strength.battle import CgBattleBackend, GameResult, run_game
from deck_strength.config import DeckStrengthSettings, GameSpec


@dataclass(slots=True)
class WorkerRuntime:
    policy: Any
    battle_backend: Any

    def run(self, game: GameSpec) -> GameResult:
        return run_game(game, self.policy, backend=self.battle_backend)


_WORKER_RUNTIME: WorkerRuntime | None = None


def build_worker_runtime(settings: DeckStrengthSettings) -> WorkerRuntime:
    torch.set_num_threads(settings.runtime.torch_threads_per_worker)
    return WorkerRuntime(
        policy=load_policy_agent(settings),
        battle_backend=CgBattleBackend(),
    )


def initialize_worker(settings: DeckStrengthSettings) -> None:
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = build_worker_runtime(settings)


def clear_worker_runtime() -> None:
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = None


def run_worker_game(game: GameSpec) -> GameResult:
    if _WORKER_RUNTIME is None:
        raise RuntimeError("Deck-strength worker is not initialized")
    return _WORKER_RUNTIME.run(game)


def collect_bounded_results(
    executor: Executor,
    games: Iterable[GameSpec],
    worker_function: Callable[[GameSpec], GameResult],
    *,
    max_pending: int,
    on_result: Callable[[GameResult], None] | None = None,
) -> list[GameResult]:
    game_iter = iter(games)
    pending = set()
    results: list[GameResult] = []

    def submit_next() -> bool:
        try:
            game = next(game_iter)
        except StopIteration:
            return False
        pending.add(executor.submit(worker_function, game))
        return True

    for _ in range(max_pending):
        if not submit_next():
            break

    while pending:
        completed, pending_now = wait(pending, return_when=FIRST_COMPLETED)
        pending = set(pending_now)
        for future in completed:
            result = future.result()
            results.append(result)
            if on_result is not None:
                on_result(result)
            submit_next()

    return sorted(results, key=lambda item: item.game_id)


def execute_games(
    settings: DeckStrengthSettings,
    games: Iterable[GameSpec],
    *,
    on_result: Callable[[GameResult], None] | None = None,
) -> list[GameResult]:
    if settings.runtime.workers == 1:
        runtime = build_worker_runtime(settings)
        results = []
        for game in games:
            result = runtime.run(game)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return sorted(results, key=lambda item: item.game_id)

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=settings.runtime.workers,
        mp_context=context,
        initializer=initialize_worker,
        initargs=(settings,),
    ) as executor:
        return collect_bounded_results(
            executor,
            games,
            run_worker_game,
            max_pending=settings.runtime.workers * 2,
            on_result=on_result,
        )
