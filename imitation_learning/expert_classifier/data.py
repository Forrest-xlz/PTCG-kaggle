"""Date selection and replay-grouped labels for expert classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Iterable, Mapping

import numpy as np

from training.feature_cache import (
    PLAYER_RESULT_WIN,
    MmapFeatureDataset,
    _mix_episode_keys,
)


Date = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ClassifierSplit:
    train_indices: np.ndarray
    train_labels: np.ndarray
    validation_indices: np.ndarray
    validation_labels: np.ndarray


def select_recent_dates(
    dates: Iterable[Date], recent_dates: int | None
) -> tuple[Date, ...]:
    ordered = tuple(sorted(set(dates)))
    if not ordered:
        raise ValueError("no cache dates are available")
    if recent_dates is None:
        return ordered
    if type(recent_dates) is not int or recent_dates < 1:
        raise ValueError("recent_dates must be null or an integer >= 1")
    return ordered[-recent_dates:]


def build_classifier_indices(
    dataset: MmapFeatureDataset,
    dates: Iterable[Date],
    expert_episode_ids: Mapping[Date, AbstractSet[int]],
    validation_ratio: float,
    seed: int,
) -> ClassifierSplit:
    """Label winners and split complete replays between train and validation."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be strictly between 0 and 1")
    selected_dates = set(dates)
    missing = selected_dates - set(expert_episode_ids)
    if missing:
        raise ValueError(f"expert episode IDs missing for dates: {sorted(missing)}")
    threshold = int(validation_ratio * (1 << 32))
    index_dtype = (
        np.uint32
        if len(dataset) <= np.iinfo(np.uint32).max
        else np.uint64
    )
    train_ids: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    validation_ids: list[np.ndarray] = []
    validation_labels: list[np.ndarray] = []

    for shard_id, (shard, date) in enumerate(
        zip(dataset.shards, dataset.shard_dates)
    ):
        if date not in selected_dates:
            continue
        winners = shard.arrays["player_result"] == PLAYER_RESULT_WIN
        ids = np.fromiter(expert_episode_ids[date], dtype=np.uint64)
        labels = np.isin(shard.arrays["episode_id"], ids)
        mixed = _mix_episode_keys(
            shard.arrays["episode_key"],
            int(seed) ^ (date[0] << 16) ^ date[1],
        ).astype(np.uint64)
        validation = winners & (mixed < threshold)
        training = winners & ~validation
        global_ids = np.arange(
            dataset.starts[shard_id],
            dataset.ends[shard_id],
            dtype=index_dtype,
        )
        train_ids.append(global_ids[training])
        train_labels.append(labels[training].astype(np.float32))
        validation_ids.append(global_ids[validation])
        validation_labels.append(labels[validation].astype(np.float32))

    def combine(parts: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
        return np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype=dtype)

    result = ClassifierSplit(
        train_indices=combine(train_ids, index_dtype),
        train_labels=combine(train_labels, np.dtype(np.float32)),
        validation_indices=combine(validation_ids, index_dtype),
        validation_labels=combine(validation_labels, np.dtype(np.float32)),
    )
    if result.train_indices.size == 0 or result.validation_indices.size == 0:
        raise ValueError("classifier train/validation split is empty")
    if not np.any(result.train_labels == 1):
        raise ValueError("classifier training split contains no expert samples")
    if not np.any(result.train_labels == 0):
        raise ValueError("classifier training split contains no non-expert samples")
    return result
