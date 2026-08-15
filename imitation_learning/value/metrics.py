"""Exact streaming regression metrics and value validation."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from training.feature_cache import EncoderBatch, IndexBatch
from value.data import player_result_targets


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    rmse: float
    explained_variance: float
    samples: int
    target_mean: float
    prediction_mean: float


@dataclass(frozen=True, slots=True)
class ValueValidationResult:
    overall: RegressionMetrics
    subgroups: dict[str, RegressionMetrics]
    seconds: float


class RegressionAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.target_sum = 0.0
        self.target_square_sum = 0.0
        self.prediction_sum = 0.0
        self.error_sum = 0.0
        self.error_square_sum = 0.0

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        return np.asarray(value, dtype=np.float64).reshape(-1)

    def update(self, predictions: Any, targets: Any) -> None:
        prediction = self._array(predictions)
        target = self._array(targets)
        if prediction.shape != target.shape:
            raise ValueError("predictions and targets must have equal shape")
        error = target - prediction
        self.count += int(target.size)
        self.target_sum += float(target.sum())
        self.target_square_sum += float(np.square(target).sum())
        self.prediction_sum += float(prediction.sum())
        self.error_sum += float(error.sum())
        self.error_square_sum += float(np.square(error).sum())

    def finalize(self) -> RegressionMetrics:
        if self.count < 1:
            raise ValueError("regression metric subset is empty")
        target_mean = self.target_sum / self.count
        prediction_mean = self.prediction_sum / self.count
        error_mean = self.error_sum / self.count
        target_variance = max(
            0.0,
            self.target_square_sum / self.count - target_mean * target_mean,
        )
        error_variance = max(
            0.0,
            self.error_square_sum / self.count - error_mean * error_mean,
        )
        explained = (
            float("nan")
            if target_variance == 0.0
            else 1.0 - error_variance / target_variance
        )
        return RegressionMetrics(
            rmse=math.sqrt(self.error_square_sum / self.count),
            explained_variance=explained,
            samples=self.count,
            target_mean=target_mean,
            prediction_mean=prediction_mean,
        )


def _to_device(
    array: np.ndarray, device: torch.device, dtype=None
) -> torch.Tensor:
    return torch.from_numpy(array).to(
        device=device,
        dtype=dtype,
        non_blocking=device.type == "cuda",
    )


def forward_encoder_batch(
    batch: EncoderBatch,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = model(
        _to_device(batch.encoder_index, device, dtype=torch.long),
        _to_device(batch.encoder_value, device, dtype=torch.float32),
        _to_device(batch.encoder_offset, device, dtype=torch.long),
        _to_device(batch.encoder_pokemon_appear, device, dtype=torch.long),
        _to_device(batch.own_summary, device, dtype=torch.float32),
        _to_device(batch.opponent_summary, device, dtype=torch.float32),
        _to_device(batch.global_summary, device, dtype=torch.float32),
        _to_device(batch.history_select_type, device, dtype=torch.long),
        _to_device(batch.history_select_context, device, dtype=torch.long),
        _to_device(batch.history_valid, device, dtype=torch.long),
        _to_device(
            batch.history_option_categorical, device, dtype=torch.long
        ),
        _to_device(batch.history_structural, device, dtype=torch.long),
        _to_device(
            batch.history_pokemon_dynamic, device, dtype=torch.float32
        ),
        _to_device(
            batch.history_attack_dynamic, device, dtype=torch.float32
        ),
        _to_device(batch.history_option_offset, device, dtype=torch.long),
    )
    targets = _to_device(
        player_result_targets(batch.player_result),
        device,
        dtype=torch.float32,
    )
    return predictions, targets


def evaluate_value_dataset(
    *,
    model: torch.nn.Module,
    dataset,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    precision,
    subgroup_masks: dict[str, np.ndarray],
) -> ValueValidationResult:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("validation indices must be a non-empty vector")
    masks = {
        name: np.asarray(mask, dtype=np.bool_)
        for name, mask in subgroup_masks.items()
    }
    for name, mask in masks.items():
        if mask.ndim != 1 or len(mask) != len(indices):
            raise ValueError(f"{name} mask must align with validation indices")
        if not np.any(mask):
            raise ValueError(f"{name} validation subset is empty")

    overall = RegressionAccumulator()
    subgroup_totals = {name: RegressionAccumulator() for name in masks}
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                end = min(start + batch_size, len(indices))
                batch = dataset.collate_encoder(IndexBatch(indices[start:end]))
                with precision.autocast():
                    predictions, targets = forward_encoder_batch(
                        batch, model, device
                    )
                overall.update(predictions, targets)
                for name, mask in masks.items():
                    local = torch.from_numpy(mask[start:end]).to(
                        device=predictions.device
                    )
                    if bool(local.any()):
                        subgroup_totals[name].update(
                            predictions[local], targets[local]
                        )
    finally:
        model.train(was_training)
    return ValueValidationResult(
        overall=overall.finalize(),
        subgroups={
            name: accumulator.finalize()
            for name, accumulator in subgroup_totals.items()
        },
        seconds=time.perf_counter() - started,
    )
