"""Train the pure behavior-cloning policy on all eligible cached samples."""
from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "cfg" / "train.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Training config not found: {CONFIG_PATH}")
with CONFIG_PATH.open("r", encoding="utf-8") as _config_handle:
    _bootstrap_config = yaml.safe_load(_config_handle) or {}
_cg_value = _bootstrap_config.get("train", {}).get("cg_path")
if not _cg_value:
    raise ValueError("train.cg_path is required in cfg/train.yaml")
_cg_path = Path(_cg_value)
if not _cg_path.is_absolute():
    _cg_path = (PROJECT_ROOT / _cg_path).resolve()
if _cg_path.name == "cg":
    _cg_path = _cg_path.parent
if not (_cg_path / "cg" / "__init__.py").exists():
    raise FileNotFoundError(
        f"train.cg_path must contain the cg package; not found under: {_cg_path}"
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(_cg_path) not in sys.path:
    sys.path.insert(0, str(_cg_path))

from cg.api import all_attack, all_card_data
from model.attack_features import build_attack_feature_table
from model.card_features import build_card_feature_table
from model.network import ModelConfig, PTCGTransformer
from training.expert_validation import ExpertLoserDateInfo, load_expert_loser_date_info
from training.feature_cache import (
    CACHE_SCHEMA_VERSION,
    ENCODER_WORDS,
    HISTORY_STEPS,
    HISTORY_STRUCTURAL_DIM,
    MAX_ACTIONS,
    ATTACK_DYNAMIC_DIM,
    OPTION_CATEGORICAL_DIM,
    OPTION_NUMERIC_DIM,
    POKEMON_DYNAMIC_DIM,
    CachedBatch,
    MmapFeatureDataset,
    LoserAugmentationCounts,
)
from training.precision import PrecisionContext


@dataclass(frozen=True)
class LoserAugmentationSettings:
    enabled: bool
    recent_dates: int
    expert_ratio: float


@dataclass(frozen=True)
class TrainSettings:
    cg_path: str
    data: str
    replay_episodes: str
    output: str
    resume: bool
    resume_checkpoint: str | None
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_steps: int
    max_samples: int | None
    seed: int
    device: str
    precision: str
    log_every_steps: int
    save_every_steps: int
    save_every_epoch: bool
    ema_alpha: float
    loser_augmentation: LoserAugmentationSettings
    train_replay_ratio: float
    train_replay_seed: int
    grad_clip_norm: float


@dataclass(frozen=True)
class ModelSettings:
    d_model: int
    ffn_multiplier: int
    num_heads: int
    encoder_layers: int
    decoder_layers: int
    norm_mode: str
    summary_mlp_layers: int
    card_mlp_layers: int
    option_numeric_mlp_layers: int
    option_token_mlp_layers: int
    transformer_activation: str = "relu"
    transformer_dropout: float = 0.0
    dropout_embedding: bool = False
    dropout_attention_probs: bool = False
    dropout_attention_output: bool = False
    dropout_ffn_output: bool = False
    card_mlp_scope: str = "shared"
    pokemon_appear_embedding: bool = False
    bench_token_mlp_layers: int = 0
    active_token_mlp_layers: int = 0
    discard_token_mlp_layers: int = 0
    hand_token_mlp_layers: int = 0
    deck_token_mlp_layers: int = 0
    region_token_mlp_residual: bool = True
    history_encoding: str = "off"
    history_action_mlp_layers: int = 1
    history_sequence_mlp_layers: int = 2


@dataclass(frozen=True)
class WandbSettings:
    enabled: bool
    project: str
    group: str
    name: str
    mode: str = "online"


@dataclass(frozen=True)
class ExperimentSettings:
    version_name: str
    train: TrainSettings
    model: ModelSettings
    wandb: WandbSettings


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
class BatchResult:
    metrics: PolicyMetrics
    grad_norm: float
    learning_rate: float
    optimizer_stepped: bool
    grad_scale: float


@dataclass(frozen=True, slots=True)
class ResumeState:
    start_epoch_index: int
    global_step: int
    history: list[dict]


class ExponentialMovingAverage:
    def __init__(self, alpha: float):
        if not 0 <= alpha < 1:
            raise ValueError("EMA alpha must be in [0, 1)")
        self.alpha = float(alpha)
        self.value: float | None = None

    def update(self, value: float) -> float:
        value = float(value)
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * value
        return self.value

    def state_dict(self) -> dict:
        return {"alpha": self.alpha, "value": self.value}

    def load_state_dict(self, state: dict) -> None:
        alpha = float(state["alpha"])
        if not math.isclose(alpha, self.alpha):
            raise ValueError(
                f"EMA alpha mismatch: checkpoint={alpha}, config={self.alpha}"
            )
        value = state.get("value")
        self.value = None if value is None else float(value)


def select_loser_augmentation_dates(
    shard_dates: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    recent_dates: int,
) -> tuple[tuple[int, int], ...]:
    dates = sorted(set(shard_dates))
    if not dates:
        raise ValueError("cache contains no replay dates")
    if len(dates) < recent_dates:
        raise ValueError(
            "loser augmentation requested "
            f"recent_dates={recent_dates}, but cache contains only "
            f"{len(dates)} dates"
        )
    return tuple(dates[-recent_dates:])


def format_loser_augmentation_line(
    info: ExpertLoserDateInfo,
    counts: LoserAugmentationCounts,
) -> str:
    date = info.date
    return (
        f"loser_aug_date={date[0]}.{date[1]} cutoff={info.cutoff:g} "
        f"participant_scores={info.participant_count:,} "
        f"episodes={info.episode_count:,} "
        f"score_eligible_episodes={counts.score_eligible_episodes:,} "
        f"selected_train_episodes={counts.selected_train_episodes:,} "
        f"loser_samples={counts.loser_samples:,}"
    )


def load_settings(path: Path = CONFIG_PATH) -> ExperimentSettings:
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping")
    missing = {"version_name", "train", "model", "wandb"} - raw.keys()
    if missing:
        raise ValueError(f"Missing YAML sections: {sorted(missing)}")

    version_name = str(raw["version_name"]).strip()
    if not version_name:
        raise ValueError("version_name must not be empty")

    def interpolate(value):
        if isinstance(value, str):
            return value.replace("${version_name}", version_name)
        if isinstance(value, dict):
            return {key: interpolate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [interpolate(item) for item in value]
        return value

    raw = interpolate(raw)
    train_raw = dict(raw["train"])
    loser_raw = train_raw.pop("loser_augmentation", None)
    if not isinstance(loser_raw, dict):
        raise ValueError("train.loser_augmentation must be a mapping")
    try:
        loser_settings = LoserAugmentationSettings(**loser_raw)
    except TypeError as exc:
        raise ValueError(
            "train.loser_augmentation must contain enabled, recent_dates, "
            "and expert_ratio"
        ) from exc
    settings = ExperimentSettings(
        version_name=version_name,
        train=TrainSettings(
            loser_augmentation=loser_settings,
            **train_raw,
        ),
        model=ModelSettings(**raw["model"]),
        wandb=WandbSettings(**raw["wandb"]),
    )
    train, model = settings.train, settings.model
    if train.epochs < 1 or train.batch_size < 1:
        raise ValueError("train.epochs and train.batch_size must be >= 1")
    if type(train.resume) is not bool:
        raise ValueError("train.resume must be true or false")
    if train.resume:
        if not isinstance(train.resume_checkpoint, str):
            raise ValueError(
                "train.resume_checkpoint must be a path when train.resume is true"
            )
        if not train.resume_checkpoint.strip():
            raise ValueError(
                "train.resume_checkpoint must not be empty when resume is true"
            )
    elif train.resume_checkpoint is not None and not isinstance(
        train.resume_checkpoint, str
    ):
        raise ValueError("train.resume_checkpoint must be null or a path")
    if train.max_samples is not None and train.max_samples < 1:
        raise ValueError("train.max_samples must be null or >= 1")
    for name in ("log_every_steps", "save_every_steps"):
        if getattr(train, name) < 1:
            raise ValueError(f"train.{name} must be >= 1")
    if train.warmup_steps < 0:
        raise ValueError("train.warmup_steps must be >= 0")
    if train.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("train.precision must be fp32, fp16, or bf16")
    if train.learning_rate <= 0 or train.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if train.grad_clip_norm <= 0:
        raise ValueError("train.grad_clip_norm must be positive")
    if not 0 <= train.beta1 < 1 or not 0 <= train.beta2 < 1:
        raise ValueError("train.beta1 and train.beta2 must be in [0, 1)")
    if not 0 <= train.ema_alpha < 1:
        raise ValueError("train.ema_alpha must be in [0, 1)")
    loser = train.loser_augmentation
    if type(loser.enabled) is not bool:
        raise ValueError("train.loser_augmentation.enabled must be true or false")
    if type(loser.recent_dates) is not int or loser.recent_dates < 1:
        raise ValueError(
            "train.loser_augmentation.recent_dates must be an integer >= 1"
        )
    if not 0 < loser.expert_ratio <= 1:
        raise ValueError(
            "train.loser_augmentation.expert_ratio must be in (0, 1]"
        )
    if not 0 < train.train_replay_ratio <= 1:
        raise ValueError("train.train_replay_ratio must be in (0, 1]")
    if type(train.train_replay_seed) is not int:
        raise ValueError("train.train_replay_seed must be an integer")
    if not isinstance(train.save_every_epoch, bool):
        raise ValueError("train.save_every_epoch must be true or false")
    if model.d_model < 1 or model.ffn_multiplier <= 0:
        raise ValueError("model.d_model and model.ffn_multiplier must be positive")
    if model.num_heads < 1 or model.d_model % model.num_heads != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if model.encoder_layers < 1 or model.decoder_layers < 1:
        raise ValueError("encoder_layers and decoder_layers must be >= 1")
    if model.norm_mode not in {"prenorm", "postnorm"}:
        raise ValueError("model.norm_mode must be prenorm or postnorm")
    if type(model.summary_mlp_layers) is not int or model.summary_mlp_layers < 1:
        raise ValueError(
            "model.summary_mlp_layers must be an integer >= 1"
        )
    if type(model.card_mlp_layers) is not int or model.card_mlp_layers < 0:
        raise ValueError(
            "model.card_mlp_layers must be an integer >= 0"
        )
    if model.card_mlp_scope not in {"shared", "region"}:
        raise ValueError(
            "model.card_mlp_scope must be shared or region"
        )
    if type(model.pokemon_appear_embedding) is not bool:
        raise ValueError(
            "model.pokemon_appear_embedding must be true or false"
        )
    for name in (
        "bench_token_mlp_layers",
        "active_token_mlp_layers",
        "discard_token_mlp_layers",
        "hand_token_mlp_layers",
        "deck_token_mlp_layers",
    ):
        value = getattr(model, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"model.{name} must be an integer >= 0")
    if type(model.region_token_mlp_residual) is not bool:
        raise ValueError(
            "model.region_token_mlp_residual must be true or false"
        )
    if (
        type(model.option_token_mlp_layers) is not int
        or model.option_token_mlp_layers < 0
    ):
        raise ValueError(
            "model.option_token_mlp_layers must be an integer >= 0"
        )
    if (
        type(model.option_numeric_mlp_layers) is not int
        or model.option_numeric_mlp_layers < 1
    ):
        raise ValueError(
            "model.option_numeric_mlp_layers must be an integer >= 1"
        )
    if settings.wandb.enabled and not settings.wandb.project:
        raise ValueError("wandb.project is required when wandb.enabled is true")
    return settings


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])


def load_epoch_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    precision: PrecisionContext,
    ema: dict[str, ExponentialMovingAverage],
    expected_model_config: dict,
    target_epochs: int,
) -> ResumeState:
    path = Path(path)
    match = re.fullmatch(r"epoch-(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(
            "training can resume only from a completed epoch-*.pt checkpoint"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Resume checkpoint root must be a mapping")
    required = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "global_step",
        "epoch",
        "config",
        "ema",
        "history",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(
            "Checkpoint is not a complete training checkpoint; missing: "
            f"{missing}"
        )
    completed_epoch = checkpoint["epoch"]
    if type(completed_epoch) is not int or completed_epoch < 1:
        raise ValueError("checkpoint epoch must be a positive integer")
    filename_epoch = int(match.group(1))
    if filename_epoch != completed_epoch:
        raise ValueError(
            f"checkpoint filename epoch {filename_epoch} does not match "
            f"payload epoch {completed_epoch}"
        )
    if completed_epoch >= target_epochs:
        raise ValueError(
            f"checkpoint already completed epoch {completed_epoch}, but "
            f"train.epochs is {target_epochs}; set a larger total epoch count"
        )
    if checkpoint["config"] != expected_model_config:
        raise ValueError(
            "checkpoint model configuration does not match the current model"
        )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    precision.load_state_dict(checkpoint["scaler"])
    checkpoint_ema = checkpoint["ema"]
    if set(checkpoint_ema) != set(ema):
        raise ValueError("checkpoint EMA metrics do not match current metrics")
    for name, tracker in ema.items():
        tracker.load_state_dict(checkpoint_ema[name])
    history = checkpoint["history"]
    if not isinstance(history, list):
        raise ValueError("checkpoint history must be a list")
    global_step = checkpoint["global_step"]
    if type(global_step) is not int or global_step < 0:
        raise ValueError("checkpoint global_step must be a non-negative integer")
    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        _restore_rng_state(rng_state)
    else:
        print(
            "warning: checkpoint has no RNG state; resume is valid but not "
            "bit-for-bit identical to uninterrupted training",
            flush=True,
        )
    return ResumeState(
        start_epoch_index=completed_epoch,
        global_step=global_step,
        history=list(history),
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
):
    if total_steps < 1:
        raise ValueError("total training steps must be >= 1")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    decay_steps = total_steps - warmup_steps

    def lr_multiplier(step_index: int) -> float:
        if warmup_steps and step_index < warmup_steps:
            return float(step_index + 1) / float(warmup_steps)
        if decay_steps == 1:
            return 0.0
        progress = (step_index - warmup_steps) / float(decay_steps - 1)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def feature_signature(config: ModelConfig) -> dict:
    return {
        "card_count": config.card_count,
        "attack_count": config.attack_count,
        "encoder_size": config.encoder_size,
        "encoder_tokens": ENCODER_WORDS,
        "encoder_layout": "numeric-summary-26-appear-v2",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "decoder_layout": "routed-option-dynamics-plus-numeric-v7",
        "option_categorical_dim": OPTION_CATEGORICAL_DIM,
        "option_numeric_dim": OPTION_NUMERIC_DIM,
        "pokemon_dynamic_dim": POKEMON_DYNAMIC_DIM,
        "attack_dynamic_dim": ATTACK_DYNAMIC_DIM,
        "history_steps": HISTORY_STEPS,
        "history_structural_dim": HISTORY_STRUCTURAL_DIM,
        "history_layout": "selected-option-superset-v1",
        "max_actions": MAX_ACTIONS,
        "action_enumeration": "max-to-min-v1",
    }


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device selected but CUDA is unavailable")
    return device


def resolve_output_root(
    train_cfg: TrainSettings,
    wandb_run,
) -> Path:
    if wandb_run is not None:
        return Path(wandb_run.dir).parent / "local-output"
    return project_path(train_cfg.output)


def should_trigger(interval: int, global_step: int) -> bool:
    return global_step > 0 and global_step % interval == 0


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
        _to_device(
            batch.encoder_pokemon_appear,
            device,
            dtype=torch.long,
        ),
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
            (top_indices[:, :width] == targets.unsqueeze(1))
            .any(dim=1)
            .sum()
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


def train_batch(
    batch: CachedBatch,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    precision: PrecisionContext,
    grad_clip_norm: float,
) -> BatchResult:
    optimizer.zero_grad(set_to_none=True)
    learning_rate = float(optimizer.param_groups[0]["lr"])
    with precision.autocast():
        logits, targets, action_counts = _forward_batch(batch, model, device)
        loss, metrics = policy_metrics(logits, targets, action_counts)
    step = precision.backward_step(
        loss=loss,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip_norm=grad_clip_norm,
    )
    return BatchResult(
        metrics=metrics,
        grad_norm=step.grad_norm,
        learning_rate=learning_rate,
        optimizer_stepped=step.optimizer_stepped,
        grad_scale=step.grad_scale,
    )


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    precision: PrecisionContext,
    settings: ExperimentSettings,
    config: ModelConfig,
    global_step: int,
    epoch: int,
    ema: dict[str, ExponentialMovingAverage],
    history: list[dict],
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": precision.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "config": config.to_dict(),
        "experiment": {
            "version_name": settings.version_name,
            "train": asdict(settings.train),
            "model": asdict(settings.model),
            "wandb": asdict(settings.wandb),
            "cache_signature": feature_signature(config),
        },
        "ema": {name: tracker.state_dict() for name, tracker in ema.items()},
        "history": history,
        "rng_state": _capture_rng_state(),
    }


def main() -> None:
    settings = load_settings()
    train_cfg, model_cfg, wandb_cfg = (
        settings.train,
        settings.model,
        settings.wandb,
    )
    data_path = project_path(train_cfg.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training cache directory not found: {data_path}")
    replay_root = project_path(train_cfg.replay_episodes)

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    cards = all_card_data()
    attacks = all_attack()
    config = ModelConfig(
        card_count=max(card.cardId for card in cards) + 1,
        attack_count=max(attack.attackId for attack in attacks) + 1,
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        d_feedforward=int(model_cfg.d_model * model_cfg.ffn_multiplier),
        encoder_layers=model_cfg.encoder_layers,
        decoder_layers=model_cfg.decoder_layers,
        norm_mode=model_cfg.norm_mode,
        transformer_activation=model_cfg.transformer_activation,
        transformer_dropout=model_cfg.transformer_dropout,
        dropout_embedding=model_cfg.dropout_embedding,
        dropout_attention_probs=model_cfg.dropout_attention_probs,
        dropout_attention_output=model_cfg.dropout_attention_output,
        dropout_ffn_output=model_cfg.dropout_ffn_output,
        summary_mlp_layers=model_cfg.summary_mlp_layers,
        card_mlp_layers=model_cfg.card_mlp_layers,
        option_numeric_mlp_layers=model_cfg.option_numeric_mlp_layers,
        option_token_mlp_layers=model_cfg.option_token_mlp_layers,
        card_mlp_scope=model_cfg.card_mlp_scope,
        pokemon_appear_embedding=model_cfg.pokemon_appear_embedding,
        bench_token_mlp_layers=model_cfg.bench_token_mlp_layers,
        active_token_mlp_layers=model_cfg.active_token_mlp_layers,
        discard_token_mlp_layers=model_cfg.discard_token_mlp_layers,
        hand_token_mlp_layers=model_cfg.hand_token_mlp_layers,
        deck_token_mlp_layers=model_cfg.deck_token_mlp_layers,
        region_token_mlp_residual=model_cfg.region_token_mlp_residual,
        history_encoding=model_cfg.history_encoding,
        history_action_mlp_layers=model_cfg.history_action_mlp_layers,
        history_sequence_mlp_layers=model_cfg.history_sequence_mlp_layers,
    )
    device = resolve_device(train_cfg.device)
    precision = PrecisionContext(train_cfg.precision, device)
    cache_started = time.perf_counter()
    dataset = MmapFeatureDataset(
        data_path,
        expected_signature=feature_signature(config),
    )
    try:
        if train_cfg.loser_augmentation.enabled:
            loser_dates = select_loser_augmentation_dates(
                dataset.shard_dates,
                train_cfg.loser_augmentation.recent_dates,
            )
            loser_date_info = load_expert_loser_date_info(
                replay_root=replay_root,
                required_dates=loser_dates,
                ratio=train_cfg.loser_augmentation.expert_ratio,
            )
            loser_episode_keys = {
                date: info.eligible_episode_keys
                for date, info in loser_date_info.items()
            }
        else:
            loser_dates = ()
            loser_date_info = {}
            loser_episode_keys = None
        selection = dataset.build_training_indices(
            train_replay_ratio=train_cfg.train_replay_ratio,
            train_replay_seed=train_cfg.train_replay_seed,
            loser_episode_keys=loser_episode_keys,
        )
    except Exception:
        dataset.close()
        raise
    cache_open_seconds = time.perf_counter() - cache_started
    if train_cfg.loser_augmentation.enabled:
        for date in loser_dates:
            print(
                format_loser_augmentation_line(
                    loser_date_info[date],
                    selection.loser_augmentation_counts[date],
                ),
                flush=True,
            )
    else:
        print("loser_augmentation=disabled", flush=True)
    print(
        f"loser_augmentation_dates={len(loser_dates):,} "
        f"loser_augmentation_replays="
        f"{selection.loser_augmentation_replays:,} "
        f"loser_augmentation_samples="
        f"{selection.loser_augmentation_samples:,} "
        f"loser_fraction_in_train="
        f"{selection.loser_fraction_in_train:.6f}",
        flush=True,
    )
    realized_train_sample_ratio = (
        len(selection.train) / selection.eligible_train_samples
    )
    realized_train_replay_ratio = (
        selection.selected_train_replays / selection.eligible_train_replays
    )
    samples_per_epoch = len(selection.train)
    if train_cfg.max_samples is not None:
        samples_per_epoch = min(samples_per_epoch, train_cfg.max_samples)
    steps_per_epoch = math.ceil(samples_per_epoch / train_cfg.batch_size)
    total_steps = steps_per_epoch * train_cfg.epochs
    if train_cfg.warmup_steps >= total_steps:
        dataset.close()
        raise ValueError("train.warmup_steps must be smaller than total steps")

    card_feature_table = build_card_feature_table(
        cards,
        config.card_count,
    )
    attack_feature_table = build_attack_feature_table(
        attacks,
        config.attack_count,
    )
    model = PTCGTransformer(
        config,
        card_feature_table,
        attack_feature_table,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )
    scheduler = build_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=train_cfg.warmup_steps,
    )

    ema = {
        name: ExponentialMovingAverage(train_cfg.ema_alpha)
        for name in ("loss", "top1", "top3", "top5")
    }
    history: list[dict] = []
    global_step = 0
    start_epoch_index = 0
    if train_cfg.resume:
        checkpoint_path = project_path(train_cfg.resume_checkpoint or "")
        try:
            resume_state = load_epoch_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                precision=precision,
                ema=ema,
                expected_model_config=config.to_dict(),
                target_epochs=train_cfg.epochs,
            )
        except Exception:
            dataset.close()
            raise
        start_epoch_index = resume_state.start_epoch_index
        global_step = resume_state.global_step
        history = resume_state.history
        print(
            f"resumed_from={checkpoint_path} "
            f"completed_epoch={start_epoch_index} "
            f"next_epoch={start_epoch_index + 1} "
            f"optimizer_step={global_step:,}",
            flush=True,
        )

    wandb_run = None
    if wandb_cfg.enabled:
        try:
            import wandb
        except ImportError as exc:
            dataset.close()
            raise RuntimeError(
                "wandb.enabled is true but wandb is not installed; "
                "run pip install -r requirements.txt"
            ) from exc
        wandb_run = wandb.init(
            project=wandb_cfg.project,
            group=wandb_cfg.group or None,
            name=wandb_cfg.name or None,
            mode=wandb_cfg.mode,
            config={
                "version_name": settings.version_name,
                "train": asdict(train_cfg),
                "model": asdict(model_cfg),
                "network": config.to_dict(),
                "cache_signature": feature_signature(config),
            },
        )
        wandb.define_metric("optimizer_step")
        for namespace in ("train/*", "epoch/*"):
            wandb.define_metric(namespace, step_metric="optimizer_step")

    output_root = resolve_output_root(train_cfg, wandb_run)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "version_name": settings.version_name,
        "train": asdict(train_cfg),
        "model": asdict(model_cfg),
        "wandb": asdict(wandb_cfg),
    }
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False),
        encoding="utf-8",
    )

    print(
        f"version={settings.version_name} device={device} "
        f"precision={train_cfg.precision} norm={model_cfg.norm_mode} "
        f"cache_shards={len(dataset.shards)} cache_samples={len(dataset):,} "
        f"eligible_train={selection.eligible_train_samples:,} "
        f"selected_train={len(selection.train):,} "
        f"eligible_train_replays={selection.eligible_train_replays:,} "
        f"selected_train_replays={selection.selected_train_replays:,} "
        f"train_sample_ratio={realized_train_sample_ratio:.4f} "
        f"train_replay_ratio={realized_train_replay_ratio:.4f} "
        f"steps_per_epoch={steps_per_epoch:,} total_steps={total_steps:,} "
        f"warmup_steps={train_cfg.warmup_steps:,}",
        flush=True,
    )
    if wandb_run is not None:
        loser_data_metrics = {
            "data/loser_augmentation_dates": len(loser_dates),
            "data/loser_augmentation_replays": (
                selection.loser_augmentation_replays
            ),
            "data/loser_augmentation_samples": (
                selection.loser_augmentation_samples
            ),
            "data/loser_fraction_in_train": selection.loser_fraction_in_train,
        }
        for date in loser_dates:
            info = loser_date_info[date]
            counts = selection.loser_augmentation_counts[date]
            prefix = f"data/loser_aug_{date[0]}_{date[1]}"
            loser_data_metrics.update(
                {
                    f"{prefix}_cutoff": info.cutoff,
                    f"{prefix}_participant_scores": info.participant_count,
                    f"{prefix}_episodes": info.episode_count,
                    f"{prefix}_score_eligible_episodes": (
                        counts.score_eligible_episodes
                    ),
                    f"{prefix}_selected_train_episodes": (
                        counts.selected_train_episodes
                    ),
                    f"{prefix}_loser_samples": counts.loser_samples,
                }
            )
        wandb_run.log(
            {
                **loser_data_metrics,
                "data/cache_open_seconds": cache_open_seconds,
                "data/cache_shards": len(dataset.shards),
                "data/cache_samples": len(dataset),
                "data/train_samples": len(selection.train),
                "data/eligible_train_samples":
                    selection.eligible_train_samples,
                "data/eligible_train_replays":
                    selection.eligible_train_replays,
                "data/selected_train_replays":
                    selection.selected_train_replays,
                "data/realized_train_sample_ratio":
                    realized_train_sample_ratio,
                "data/realized_train_replay_ratio":
                    realized_train_replay_ratio,
                "schedule/steps_per_epoch": steps_per_epoch,
                "schedule/total_steps": total_steps,
                "schedule/warmup_steps": train_cfg.warmup_steps,
                "optimizer_step": global_step,
            }
        )

    skipped_updates = 0
    train_samples_seen = 0
    train_compute_seconds = 0.0
    total_training_start = time.perf_counter()

    def save_checkpoint(name: str, epoch: int) -> None:
        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                precision=precision,
                settings=settings,
                config=config,
                global_step=global_step,
                epoch=epoch,
                ema=ema,
                history=history,
            ),
            checkpoint_dir / name,
        )

    try:
        for epoch_index in range(start_epoch_index, train_cfg.epochs):
            epoch = epoch_index + 1
            epoch_started = time.perf_counter()
            model.train()
            epoch_loss_sum = 0.0
            epoch_top1 = epoch_top3 = epoch_top5 = epoch_samples = 0
            for batch in dataset.iter_batches(
                indices=selection.train,
                batch_size=train_cfg.batch_size,
                seed=train_cfg.seed + epoch_index,
                shuffle=True,
                max_samples=train_cfg.max_samples,
            ):
                batch_started = time.perf_counter()
                result = train_batch(
                    batch=batch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    precision=precision,
                    grad_clip_norm=train_cfg.grad_clip_norm,
                )
                train_compute_seconds += time.perf_counter() - batch_started
                metrics = result.metrics
                averages = metrics.averages()
                ema_values = {
                    "loss": ema["loss"].update(averages["loss"]),
                    "top1": ema["top1"].update(averages["top1_accuracy"]),
                    "top3": ema["top3"].update(averages["top3_accuracy"]),
                    "top5": ema["top5"].update(averages["top5_accuracy"]),
                }
                train_samples_seen += metrics.samples
                epoch_loss_sum += metrics.loss * metrics.samples
                epoch_top1 += metrics.top1_correct
                epoch_top3 += metrics.top3_correct
                epoch_top5 += metrics.top5_correct
                epoch_samples += metrics.samples

                if not result.optimizer_stepped:
                    skipped_updates += 1
                    continue
                global_step += 1

                if should_trigger(train_cfg.log_every_steps, global_step):
                    samples_per_second = train_samples_seen / max(
                        train_compute_seconds, 1e-9
                    )
                    payload = {
                        "train/ema_loss": ema_values["loss"],
                        "train/ema_top1_accuracy": ema_values["top1"],
                        "train/ema_top3_accuracy": ema_values["top3"],
                        "train/ema_top5_accuracy": ema_values["top5"],
                        "train/grad_norm": result.grad_norm,
                        "train/grad_scale": result.grad_scale,
                        "train/learning_rate": result.learning_rate,
                        "train/samples": train_samples_seen,
                        "train/samples_per_second": samples_per_second,
                        "train/epoch": epoch,
                        "train/skipped_optimizer_steps": skipped_updates,
                        "optimizer_step": global_step,
                    }
                    print(
                        f"epoch={epoch} step={global_step:,} "
                        f"samples={train_samples_seen:,} "
                        f"loss_ema={ema_values['loss']:.4f} "
                        f"top1_ema={ema_values['top1']:.3f} "
                        f"top3_ema={ema_values['top3']:.3f} "
                        f"top5_ema={ema_values['top5']:.3f} "
                        f"lr={result.learning_rate:.3e} "
                        f"samples/s={samples_per_second:.1f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(payload)

                if should_trigger(train_cfg.save_every_steps, global_step):
                    save_checkpoint(f"step-{global_step:08d}.pt", epoch)

            epoch_seconds = time.perf_counter() - epoch_started
            denominator = max(epoch_samples, 1)
            row = {
                "epoch": epoch,
                "samples": epoch_samples,
                "batch_size": train_cfg.batch_size,
                "loss": epoch_loss_sum / denominator,
                "top1_accuracy": epoch_top1 / denominator,
                "top3_accuracy": epoch_top3 / denominator,
                "top5_accuracy": epoch_top5 / denominator,
                "seconds": epoch_seconds,
                "samples_per_second": epoch_samples
                / max(epoch_seconds, 1e-9),
                "optimizer_step": global_step,
                "skipped_optimizer_steps": skipped_updates,
            }
            history.append(row)
            print(row, flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch/loss": row["loss"],
                        "epoch/top1_accuracy": row["top1_accuracy"],
                        "epoch/top3_accuracy": row["top3_accuracy"],
                        "epoch/top5_accuracy": row["top5_accuracy"],
                        "epoch/samples": row["samples"],
                        "epoch/seconds": row["seconds"],
                        "epoch/samples_per_second": row[
                            "samples_per_second"
                        ],
                        "epoch/index": epoch,
                        "epoch/skipped_optimizer_steps": skipped_updates,
                        "optimizer_step": global_step,
                    }
                )
            if train_cfg.save_every_epoch:
                save_checkpoint(f"epoch-{epoch:03d}.pt", epoch)
            (output_root / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
    finally:
        dataset.close()
        if wandb_run is not None:
            if history:
                wandb_run.summary["training/final_loss"] = history[-1][
                    "loss"
                ]
                wandb_run.summary["training/final_top1_accuracy"] = history[
                    -1
                ]["top1_accuracy"]
            wandb_run.summary["training/total_seconds"] = (
                time.perf_counter() - total_training_start
            )
            wandb_run.finish()


if __name__ == "__main__":
    main()
