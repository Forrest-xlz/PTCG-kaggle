"""YAML configuration shared by expert training, inference, and recovery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "cfg" / "expert_classifier.yaml"


@dataclass(frozen=True, slots=True)
class ModelInitSettings:
    init_mode: str
    checkpoint: str | None


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    checkpoint: str
    recent_dates: int | None
    threshold: float
    output: str


@dataclass(frozen=True, slots=True)
class RecoverySettings:
    output: str


@dataclass(frozen=True, slots=True)
class ExpertClassifierSettings:
    train_config: str
    output: str
    extracted_data: str
    recent_dates: int
    expert_ratio: float
    validation_ratio: float
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_steps: int
    device: str
    precision: str
    log_every_steps: int
    grad_clip_norm: float
    model: ModelInitSettings
    inference: InferenceSettings
    recovery: RecoverySettings


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_train_yaml(settings: ExpertClassifierSettings) -> dict[str, Any]:
    path = project_path(settings.train_config)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("train"), dict):
        raise ValueError("referenced train config must contain train and model mappings")
    if not isinstance(raw.get("model"), dict):
        raise ValueError("referenced train config must contain a model mapping")
    return raw


def load_settings(path: Path = CONFIG_PATH) -> ExpertClassifierSettings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    section = raw.get("expert_classifier") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        raise ValueError("expert_classifier.yaml must contain expert_classifier")
    values = dict(section)
    try:
        model = ModelInitSettings(**values.pop("model"))
        inference = InferenceSettings(**values.pop("inference"))
        recovery = RecoverySettings(**values.pop("recovery"))
        settings = ExpertClassifierSettings(
            model=model,
            inference=inference,
            recovery=recovery,
            **values,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid expert classifier configuration: {exc}") from exc

    if settings.recent_dates < 1:
        raise ValueError("recent_dates must be >= 1")
    if settings.inference.recent_dates is not None and settings.inference.recent_dates < 1:
        raise ValueError("inference.recent_dates must be null or >= 1")
    if not 0.0 < settings.expert_ratio <= 1.0:
        raise ValueError("expert_ratio must be in (0, 1]")
    if not 0.0 < settings.validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1)")
    if not 0.0 <= settings.inference.threshold <= 1.0:
        raise ValueError("inference.threshold must be in [0, 1]")
    if settings.model.init_mode not in {"scratch", "checkpoint"}:
        raise ValueError("model.init_mode must be scratch or checkpoint")
    if settings.model.init_mode == "checkpoint" and not settings.model.checkpoint:
        raise ValueError("model.checkpoint is required in checkpoint mode")
    if settings.epochs < 1 or settings.batch_size < 1:
        raise ValueError("epochs and batch_size must be >= 1")
    if settings.learning_rate <= 0 or settings.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if not 0 <= settings.beta1 < 1 or not 0 <= settings.beta2 < 1:
        raise ValueError("beta1 and beta2 must be in [0, 1)")
    if settings.warmup_steps < 0 or settings.log_every_steps < 1:
        raise ValueError("warmup_steps must be >= 0 and log_every_steps >= 1")
    if settings.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if settings.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive")
    return settings
