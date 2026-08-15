"""All-outcome replay splits and targets for value training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Iterable, Mapping

import numpy as np

from training.feature_cache import (
    PLAYER_RESULT_DRAW,
    PLAYER_RESULT_LOSS,
    PLAYER_RESULT_WIN,
    MmapFeatureDataset,
    _mix_episode_keys,
)


@dataclass(frozen=True, slots=True)
class ValueSplits:
    train: np.ndarray
    isolation: np.ndarray
    isolation_masks: dict[str, np.ndarray]
    in_distribution: np.ndarray
    in_distribution_masks: dict[str, np.ndarray]
    latest: np.ndarray
    latest_masks: dict[str, np.ndarray]
    latest_date: tuple[int, int]


def player_result_targets(results: np.ndarray) -> np.ndarray:
    values = np.asarray(results)
    valid = np.isin(
        values, (PLAYER_RESULT_WIN, PLAYER_RESULT_DRAW, PLAYER_RESULT_LOSS)
    )
    if not np.all(valid):
        raise ValueError("player_result contains an invalid value")
    lookup = np.zeros(4, dtype=np.float32)
    lookup[PLAYER_RESULT_WIN] = 1.0
    lookup[PLAYER_RESULT_DRAW] = 0.0
    lookup[PLAYER_RESULT_LOSS] = -1.0
    return lookup[values.astype(np.int64)]


def build_value_splits(
    dataset: MmapFeatureDataset,
    *,
    validation_ratio: float,
    validation_seed: int,
    expert_episode_keys: Mapping[
        tuple[int, int], AbstractSet[int]
    ],
    top_deck_keys: Iterable[int],
    isolation_episode_keys: Mapping[
        str, Mapping[tuple[int, int], AbstractSet[int]]
    ],
) -> ValueSplits:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be strictly between 0 and 1")
    dates = set(dataset.shard_dates)
    if dates - set(expert_episode_keys):
        raise ValueError("expert episode keys must cover every cache date")
    for name, by_date in isolation_episode_keys.items():
        if dates - set(by_date):
            raise ValueError(f"{name} isolation keys must cover every cache date")
    latest_date = max(dataset.shard_dates)
    threshold = int(validation_ratio * (1 << 32))
    deck_keys = tuple(int(key) for key in top_deck_keys)
    isolation_names = tuple(sorted(map(str, isolation_episode_keys)))

    train_parts: list[np.ndarray] = []
    isolation_parts: list[np.ndarray] = []
    in_distribution_parts: list[np.ndarray] = []
    latest_parts: list[np.ndarray] = []
    isolation_mask_parts = {name: [] for name in isolation_names}
    in_mask_parts: dict[str, list[np.ndarray]] = {
        "val_in_distribution_expert": []
    }
    latest_mask_parts: dict[str, list[np.ndarray]] = {
        "val_latest_expert": []
    }
    for index in range(1, len(deck_keys) + 1):
        in_mask_parts[f"val_in_distribution_deck{index}"] = []
        in_mask_parts[f"val_in_distribution_expert_deck{index}"] = []
        latest_mask_parts[f"val_latest_deck{index}"] = []
        latest_mask_parts[f"val_latest_expert_deck{index}"] = []

    for shard_id, shard in enumerate(dataset.shards):
        date = dataset.shard_dates[shard_id]
        global_ids = np.arange(
            dataset.starts[shard_id], dataset.ends[shard_id], dtype=np.int64
        )
        episode_keys = shard.arrays["episode_key"]
        expert = np.isin(
            episode_keys,
            np.fromiter(expert_episode_keys[date], dtype=np.uint32),
        )
        per_deck = tuple(
            shard.arrays["deck_key"] == key for key in deck_keys
        )
        namespace_masks = {
            name: np.isin(
                episode_keys,
                np.fromiter(
                    isolation_episode_keys[name][date], dtype=np.uint32
                ),
            )
            for name in isolation_names
        }
        isolation = (
            np.logical_or.reduce(tuple(namespace_masks.values()))
            if namespace_masks
            else np.zeros(len(shard), dtype=np.bool_)
        )
        if np.any(isolation):
            isolation_parts.append(global_ids[isolation])
            for name, mask in namespace_masks.items():
                isolation_mask_parts[name].append(mask[isolation])

        if date == latest_date:
            selected = ~isolation
            latest_parts.append(global_ids[selected])
            latest_mask_parts["val_latest_expert"].append(expert[selected])
            for index, deck_mask in enumerate(per_deck, 1):
                latest_mask_parts[f"val_latest_deck{index}"].append(
                    deck_mask[selected]
                )
                latest_mask_parts[f"val_latest_expert_deck{index}"].append(
                    (expert & deck_mask)[selected]
                )
            continue

        validation_replay = (
            _mix_episode_keys(episode_keys, validation_seed).astype(np.uint64)
            < threshold
        )
        validation = ~isolation & validation_replay
        training = ~isolation & ~validation_replay
        in_distribution_parts.append(global_ids[validation])
        train_parts.append(global_ids[training])
        in_mask_parts["val_in_distribution_expert"].append(
            expert[validation]
        )
        for index, deck_mask in enumerate(per_deck, 1):
            in_mask_parts[f"val_in_distribution_deck{index}"].append(
                deck_mask[validation]
            )
            in_mask_parts[
                f"val_in_distribution_expert_deck{index}"
            ].append((expert & deck_mask)[validation])

    def combine(parts: list[np.ndarray], name: str) -> np.ndarray:
        result = np.concatenate(parts) if parts else np.empty(0, np.int64)
        if result.size == 0:
            raise ValueError(f"{name} is empty")
        return result

    def combine_masks(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            name: (
                np.concatenate(values).astype(np.bool_, copy=False)
                if values
                else np.empty(0, dtype=np.bool_)
            )
            for name, values in parts.items()
        }

    return ValueSplits(
        train=combine(train_parts, "value training set"),
        isolation=combine(isolation_parts, "value isolation validation"),
        isolation_masks=combine_masks(isolation_mask_parts),
        in_distribution=combine(
            in_distribution_parts, "value in-distribution validation"
        ),
        in_distribution_masks=combine_masks(in_mask_parts),
        latest=combine(latest_parts, "value latest validation"),
        latest_masks=combine_masks(latest_mask_parts),
        latest_date=latest_date,
    )
