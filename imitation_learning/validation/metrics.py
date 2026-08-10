"""Read-only policy metrics matching the training evaluator."""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from training.feature_cache import CachedBatch, IndexBatch, MmapFeatureDataset


@dataclass(frozen=True, slots=True)
class PolicyMetrics:
    loss: float
    top1_correct: int
    top3_correct: int
    top5_correct: int
    samples: int

    def averages(self) -> dict[str, float]:
        denominator = max(self.samples, 1)
        return {
            "loss": self.loss,
            "top1_accuracy": self.top1_correct / denominator,
            "top3_accuracy": self.top3_correct / denominator,
            "top5_accuracy": self.top5_correct / denominator,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    overall: PolicyMetrics
    subgroups: dict[str, PolicyMetrics]
    seconds: float


def _to_device(
    array: np.ndarray,
    device: torch.device,
    dtype=None,
) -> torch.Tensor:
    return torch.from_numpy(array).to(
        device=device,
        dtype=dtype,
        non_blocking=device.type == "cuda",
    )


def _forward_batch(
    batch: CachedBatch,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model(
        _to_device(batch.encoder_index, device),
        _to_device(batch.encoder_value, device, dtype=torch.float32),
        _to_device(batch.encoder_offset, device),
        _to_device(batch.encoder_pokemon_appear, device, dtype=torch.long),
        _to_device(batch.own_summary, device, dtype=torch.float32),
        _to_device(batch.opponent_summary, device, dtype=torch.float32),
        _to_device(batch.global_summary, device, dtype=torch.float32),
        _to_device(batch.history_select_type, device, dtype=torch.long),
        _to_device(batch.history_select_context, device, dtype=torch.long),
        _to_device(batch.history_valid, device, dtype=torch.long),
        _to_device(batch.history_option_categorical, device, dtype=torch.long),
        _to_device(batch.history_structural, device, dtype=torch.long),
        _to_device(batch.history_pokemon_dynamic, device, dtype=torch.float32),
        _to_device(batch.history_attack_dynamic, device, dtype=torch.float32),
        _to_device(batch.history_option_offset, device, dtype=torch.long),
        _to_device(batch.option_categorical, device, dtype=torch.long),
        _to_device(batch.option_numeric, device, dtype=torch.float32),
        _to_device(batch.pokemon_dynamic, device, dtype=torch.float32),
        _to_device(batch.attack_dynamic, device, dtype=torch.float32),
        _to_device(batch.action_option_index, device, dtype=torch.long),
        _to_device(batch.action_option_offset, device, dtype=torch.long),
    )
    targets = _to_device(batch.target, device, dtype=torch.long)
    action_counts = _to_device(batch.action_count, device, dtype=torch.long)
    return logits, targets, action_counts


def policy_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    action_counts: torch.Tensor,
) -> tuple[torch.Tensor, PolicyMetrics]:
    invalid = (
        torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
        >= action_counts.unsqueeze(1)
    )
    masked = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
    loss = torch.nn.functional.cross_entropy(masked, targets)
    top_indices = masked.topk(min(5, masked.shape[1]), dim=1).indices

    def correct_at(k: int) -> torch.Tensor:
        width = min(k, top_indices.shape[1])
        return (
            (top_indices[:, :width] == targets.unsqueeze(1)).any(dim=1).sum()
        )

    top1, top3, top5 = [
        int(value)
        for value in torch.stack(
            [correct_at(1), correct_at(3), correct_at(5)]
        ).tolist()
    ]
    return loss, PolicyMetrics(
        loss=float(loss.item()),
        top1_correct=top1,
        top3_correct=top3,
        top5_correct=top5,
        samples=int(targets.numel()),
    )


def legal_action_probabilities(
    logits: torch.Tensor,
    action_counts: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 softmax probabilities with invalid actions set to zero."""
    logits = logits.float()
    invalid = (
        torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
        >= action_counts.unsqueeze(1)
    )
    return torch.softmax(logits.masked_fill(invalid, float("-inf")), dim=1)


def evaluate_dataset(
    models: Sequence[torch.nn.Module],
    ensemble_enabled: bool,
    dataset: MmapFeatureDataset,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    precision,
    subgroup_masks: dict[str, np.ndarray],
) -> ValidationResult:
    models = tuple(models)
    if not models:
        raise ValueError("at least one validation model is required")
    if ensemble_enabled and len(models) < 2:
        raise ValueError("enabled ensemble requires at least two models")
    if not ensemble_enabled and len(models) != 1:
        raise ValueError("disabled ensemble requires exactly one model")
    indices = np.asarray(indices)
    if indices.ndim != 1:
        raise ValueError("validation indices must be one-dimensional")
    masks = {
        name: np.asarray(mask, dtype=np.bool_)
        for name, mask in subgroup_masks.items()
    }
    for name, mask in masks.items():
        if mask.ndim != 1 or len(mask) != len(indices):
            raise ValueError(
                f"{name} mask must be one-dimensional and align with indices"
            )
        if not np.any(mask):
            raise ValueError(f"{name} validation subset is empty")

    totals = {"overall": [0.0, 0, 0, 0, 0]}
    totals.update({name: [0.0, 0, 0, 0, 0] for name in masks})

    def accumulate(name: str, metrics: PolicyMetrics) -> None:
        total = totals[name]
        total[0] += metrics.loss * metrics.samples
        total[1] += metrics.top1_correct
        total[2] += metrics.top3_correct
        total[3] += metrics.top5_correct
        total[4] += metrics.samples

    def finalize(name: str) -> PolicyMetrics:
        loss_sum, top1, top3, top5, samples = totals[name]
        if samples < 1:
            raise ValueError(f"{name} validation subset is empty")
        return PolicyMetrics(
            loss=loss_sum / samples,
            top1_correct=top1,
            top3_correct=top3,
            top5_correct=top5,
            samples=samples,
        )

    training_states = tuple(model.training for model in models)
    for model in models:
        model.eval()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                end = min(start + batch_size, len(indices))
                batch = dataset.collate(IndexBatch(indices[start:end]))
                with precision.autocast():
                    logits, targets, action_counts = _forward_batch(
                        batch, models[0], device
                    )
                    if ensemble_enabled:
                        probability_sum = legal_action_probabilities(
                            logits, action_counts
                        )
                        for model in models[1:]:
                            model_logits, _, _ = _forward_batch(
                                batch, model, device
                            )
                            if model_logits.shape != logits.shape:
                                raise ValueError(
                                    "ensemble models produced different "
                                    "action-logit shapes"
                                )
                            probability_sum.add_(
                                legal_action_probabilities(
                                    model_logits, action_counts
                                )
                            )
                        averaged = probability_sum / len(models)
                        scores = torch.log(
                            averaged.clamp_min(
                                torch.finfo(torch.float32).tiny
                            )
                        )
                    else:
                        scores = logits
                    _, metrics = policy_metrics(
                        scores, targets, action_counts
                    )
                    accumulate("overall", metrics)
                    for name, mask in masks.items():
                        selected = torch.from_numpy(mask[start:end]).to(
                            device=device
                        )
                        if bool(selected.any()):
                            _, subgroup_metrics = policy_metrics(
                                scores[selected],
                                targets[selected],
                                action_counts[selected],
                            )
                            accumulate(name, subgroup_metrics)
    finally:
        for model, was_training in zip(models, training_states):
            model.train(was_training)
    return ValidationResult(
        overall=finalize("overall"),
        subgroups={name: finalize(name) for name in masks},
        seconds=time.perf_counter() - started,
    )


def top_deck_subgroup_masks(
    scope: str,
    deck_masks: tuple[np.ndarray, ...],
    expert_deck_masks: tuple[np.ndarray, ...],
) -> dict[str, np.ndarray]:
    if len(deck_masks) != len(expert_deck_masks):
        raise ValueError("top-deck and expert top-deck masks must align")
    result: dict[str, np.ndarray] = {}
    for deck_index, (deck_mask, expert_mask) in enumerate(
        zip(deck_masks, expert_deck_masks), start=1
    ):
        result[f"{scope}_deck{deck_index}"] = deck_mask
        result[f"{scope}_expert_deck{deck_index}"] = expert_mask
    return result


def format_metrics(
    namespace: str,
    metrics: PolicyMetrics,
    seconds: float | None = None,
) -> str:
    averages = metrics.averages()
    line = (
        f"{namespace} samples={metrics.samples:,} "
        f"loss={averages['loss']:.4f} "
        f"top1={averages['top1_accuracy']:.3f} "
        f"top3={averages['top3_accuracy']:.3f} "
        f"top5={averages['top5_accuracy']:.3f}"
    )
    if seconds is not None:
        line += f" seconds={seconds:.3f}"
    return line
