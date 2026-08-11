from __future__ import annotations

import math
from dataclasses import dataclass
from typing import AbstractSet, Mapping, Protocol, Sequence

import numpy as np

from training.feature_cache import PLAYER_RESULT_WIN


class SamplingSettingsLike(Protocol):
    base_sample_ratio: float
    expert_extra_weight: float
    deck_extra_weights: Mapping[int, float]
    expert_deck_extra_weights: Mapping[int, float]


@dataclass(frozen=True, slots=True)
class SamplingRule:
    name: str
    weight: float
    indices: np.ndarray

    @property
    def eligible_samples(self) -> int:
        return int(self.indices.size)

    @property
    def added_samples(self) -> int:
        whole = math.floor(self.weight)
        fraction = self.weight - whole
        return whole * self.eligible_samples + math.floor(
            fraction * self.eligible_samples
        )


@dataclass(frozen=True, slots=True)
class TrainingSamplingPlan:
    base_indices: np.ndarray
    base_sample_ratio: float
    rules: tuple[SamplingRule, ...]

    @property
    def base_samples(self) -> int:
        return math.floor(self.base_sample_ratio * self.base_indices.size)


def epoch_sample_count(plan: TrainingSamplingPlan) -> int:
    return plan.base_samples + sum(rule.added_samples for rule in plan.rules)


def _active_rule_specs(
    settings: SamplingSettingsLike,
) -> list[tuple[str, float, str, int | None]]:
    specs: list[tuple[str, float, str, int | None]] = []
    if settings.expert_extra_weight > 0:
        specs.append(("expert", settings.expert_extra_weight, "expert", None))
    for deck_number, weight in sorted(settings.deck_extra_weights.items()):
        if weight > 0:
            specs.append((f"deck{deck_number}", weight, "deck", deck_number))
    for deck_number, weight in sorted(
        settings.expert_deck_extra_weights.items()
    ):
        if weight > 0:
            specs.append(
                (
                    f"expert_deck{deck_number}",
                    weight,
                    "expert_deck",
                    deck_number,
                )
            )
    return specs


def build_sampling_plan(
    dataset,
    train_indices: np.ndarray,
    expert_episode_keys: Mapping[tuple[int, int], AbstractSet[int]] | None,
    top_deck_keys: Sequence[int],
    settings: SamplingSettingsLike,
) -> TrainingSamplingPlan:
    indices = np.asarray(train_indices)
    if (
        indices.ndim != 1
        or indices.size == 0
        or np.any(indices[1:] <= indices[:-1])
    ):
        raise ValueError(
            "training indices must be a nonempty sorted one-dimensional vector"
        )

    specs = _active_rule_specs(settings)
    needs_expert = any(
        kind in {"expert", "expert_deck"}
        for _, _, kind, _ in specs
    )
    if needs_expert:
        if expert_episode_keys is None:
            raise ValueError(
                "expert episode keys are required by active expert rules"
            )
        missing_dates = set(dataset.shard_dates) - set(expert_episode_keys)
        if missing_dates:
            raise ValueError(
                f"expert episode keys are missing dates: {sorted(missing_dates)}"
            )

    collected: dict[str, list[np.ndarray]] = {name: [] for name, *_ in specs}
    for shard, start, end, date in zip(
        dataset.shards,
        dataset.starts,
        dataset.ends,
        dataset.shard_dates,
    ):
        left = int(np.searchsorted(indices, start, side="left"))
        right = int(np.searchsorted(indices, end, side="left"))
        shard_global = indices[left:right]
        if shard_global.size == 0:
            continue
        local = shard_global - int(start)
        winner = shard.arrays["player_result"][local] == PLAYER_RESULT_WIN

        expert = None
        if needs_expert:
            date_keys = np.fromiter(
                expert_episode_keys[date], dtype=np.uint32
            )
            expert = np.isin(
                shard.arrays["episode_key"][local],
                date_keys,
                assume_unique=False,
            )

        for name, _, kind, deck_number in specs:
            mask = winner.copy()
            if kind in {"expert", "expert_deck"}:
                mask &= expert
            if kind in {"deck", "expert_deck"}:
                deck_key = int(top_deck_keys[int(deck_number) - 1])
                mask &= shard.arrays["deck_key"][local] == deck_key
            if np.any(mask):
                collected[name].append(shard_global[mask])

    rules: list[SamplingRule] = []
    for name, weight, _, _ in specs:
        parts = collected[name]
        if not parts:
            raise ValueError(
                f"sampling rule {name} has no eligible winning samples"
            )
        rule_indices = np.concatenate(parts).astype(indices.dtype, copy=False)
        rules.append(SamplingRule(name, float(weight), rule_indices))

    return TrainingSamplingPlan(
        base_indices=indices,
        base_sample_ratio=float(settings.base_sample_ratio),
        rules=tuple(rules),
    )


def build_epoch_indices(plan: TrainingSamplingPlan, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []

    if plan.base_samples == plan.base_indices.size:
        parts.append(plan.base_indices)
    elif plan.base_samples:
        parts.append(
            rng.choice(
                plan.base_indices,
                size=plan.base_samples,
                replace=False,
            )
        )

    for rule in plan.rules:
        whole = math.floor(rule.weight)
        parts.extend([rule.indices] * whole)
        fractional_count = rule.added_samples - whole * rule.eligible_samples
        if fractional_count:
            parts.append(
                rng.choice(rule.indices, size=fractional_count, replace=False)
            )

    if not parts:
        return np.empty(0, dtype=plan.base_indices.dtype)
    result = np.concatenate(parts).astype(plan.base_indices.dtype, copy=False)
    rng.shuffle(result)
    return result
