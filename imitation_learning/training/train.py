"""Train the pure-BC policy from packed mmap feature caches."""
from __future__ import annotations

import json
import math
import random
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

from cg.api import SelectContext, all_attack, all_card_data
from model.network import ModelConfig, PTCGTransformer
from training.feature_cache import MAX_ACTIONS, CachedBatch, MmapFeatureDataset
from training.precision import PrecisionContext


@dataclass(frozen=True)
class TrainSettings:
    cg_path: str
    data: str
    output: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_ratio: float
    max_samples: int | None
    seed: int
    device: str
    precision: str
    shuffle_mode: str
    log_every_steps: int
    grad_clip_norm: float


@dataclass(frozen=True)
class ModelSettings:
    d_model: int
    ffn_multiplier: int
    num_heads: int
    encoder_layers: int
    decoder_layers: int


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
class BatchResult:
    loss: float
    correct: int
    grad_norm: float
    learning_rate: float
    optimizer_stepped: bool
    grad_scale: float


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
    settings = ExperimentSettings(
        version_name=version_name,
        train=TrainSettings(**raw["train"]),
        model=ModelSettings(**raw["model"]),
        wandb=WandbSettings(**raw["wandb"]),
    )
    train, model = settings.train, settings.model
    if train.epochs < 1 or train.batch_size < 1:
        raise ValueError("train.epochs and train.batch_size must be >= 1")
    if train.max_samples is not None and train.max_samples < 1:
        raise ValueError("train.max_samples must be null or >= 1")
    if train.log_every_steps < 1:
        raise ValueError("train.log_every_steps must be >= 1")
    if train.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("train.precision must be fp32, fp16, or bf16")
    if train.shuffle_mode not in {"global", "shard"}:
        raise ValueError("train.shuffle_mode must be global or shard")
    if train.learning_rate <= 0 or train.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if train.grad_clip_norm <= 0:
        raise ValueError("train.grad_clip_norm must be positive")
    if not 0 <= train.warmup_ratio < 1:
        raise ValueError("train.warmup_ratio must be in [0, 1)")
    if not 0 <= train.beta1 < 1 or not 0 <= train.beta2 < 1:
        raise ValueError("train.beta1 and train.beta2 must be in [0, 1)")
    if model.d_model < 1 or model.ffn_multiplier <= 0:
        raise ValueError("model.d_model and model.ffn_multiplier must be positive")
    if model.num_heads < 1 or model.d_model % model.num_heads != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if model.encoder_layers < 1 or model.decoder_layers < 1:
        raise ValueError("encoder_layers and decoder_layers must be >= 1")
    if settings.wandb.enabled and not settings.wandb.project:
        raise ValueError("wandb.project is required when wandb.enabled is true")
    return settings


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_lr_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    if total_steps < 1:
        raise ValueError("total training steps must be >= 1")
    warmup_steps = min(total_steps - 1, round(total_steps * warmup_ratio))

    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier), warmup_steps


def feature_signature(config: ModelConfig) -> dict:
    return {
        "card_count": config.card_count,
        "attack_count": config.attack_count,
        "encoder_size": config.encoder_size,
        "num_encoder_words": config.num_encoder_words,
        "decoder_size": config.decoder_size,
        "recover_special_condition": config.recover_special_condition,
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


def _to_device(array: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    return tensor.to(
        device=device,
        dtype=dtype,
        non_blocking=device.type == "cuda",
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
    encoder_index = _to_device(batch.encoder_index, device)
    encoder_value = _to_device(batch.encoder_value, device, dtype=torch.float32)
    encoder_offset = _to_device(batch.encoder_offset, device)
    decoder_index = _to_device(batch.decoder_index, device)
    decoder_offset = _to_device(batch.decoder_offset, device)
    targets = _to_device(batch.target, device, dtype=torch.long)
    action_counts = _to_device(batch.action_count, device, dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    learning_rate = float(optimizer.param_groups[0]["lr"])
    with precision.autocast():
        policy_logits = model(
            encoder_index,
            encoder_value,
            encoder_offset,
            decoder_index,
            decoder_offset,
        )
        invalid_actions = (
            torch.arange(MAX_ACTIONS, device=device).unsqueeze(0)
            >= action_counts.unsqueeze(1)
        )
        policy_logits = policy_logits.masked_fill(
            invalid_actions, torch.finfo(policy_logits.dtype).min
        )
        loss = torch.nn.functional.cross_entropy(policy_logits, targets)

    step = precision.backward_step(
        loss=loss,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip_norm=grad_clip_norm,
    )
    correct = int((policy_logits.argmax(dim=1) == targets).sum().item())
    return BatchResult(
        loss=float(loss.item()),
        correct=correct,
        grad_norm=step.grad_norm,
        learning_rate=learning_rate,
        optimizer_stepped=step.optimizer_stepped,
        grad_scale=step.grad_scale,
    )


def main() -> None:
    settings = load_settings()
    train_cfg, model_cfg, wandb_cfg = (
        settings.train,
        settings.model,
        settings.wandb,
    )
    data_path = project_path(train_cfg.data)
    output_path = project_path(train_cfg.output)
    if not data_path.exists():
        raise FileNotFoundError(f"Training cache directory not found: {data_path}")

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    output_path.mkdir(parents=True, exist_ok=True)

    cards = all_card_data()
    config = ModelConfig(
        card_count=max(card.cardId for card in cards) + 1,
        attack_count=max(attack.attackId for attack in all_attack()) + 1,
        recover_special_condition=int(SelectContext.RECOVER_SPECIAL_CONDITION),
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        d_feedforward=int(model_cfg.d_model * model_cfg.ffn_multiplier),
        encoder_layers=model_cfg.encoder_layers,
        decoder_layers=model_cfg.decoder_layers,
    )
    device = resolve_device(train_cfg.device)
    precision = PrecisionContext(train_cfg.precision, device)
    cache_start = time.perf_counter()
    dataset = MmapFeatureDataset(
        data_path,
        expected_signature=feature_signature(config),
    )
    cache_open_seconds = time.perf_counter() - cache_start
    samples_per_epoch = len(dataset)
    if train_cfg.max_samples is not None:
        samples_per_epoch = min(samples_per_epoch, train_cfg.max_samples)
    steps_per_epoch = math.ceil(samples_per_epoch / train_cfg.batch_size)
    total_steps = steps_per_epoch * train_cfg.epochs

    model = PTCGTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )
    scheduler, warmup_steps = build_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=train_cfg.warmup_ratio,
    )
    history = []
    print(
        f"version={settings.version_name} device={device} "
        f"precision={train_cfg.precision} shuffle={train_cfg.shuffle_mode} "
        f"cache_shards={len(dataset.shards)} cache_samples={len(dataset):,} "
        f"samples_per_epoch={samples_per_epoch:,} "
        f"batch_size={train_cfg.batch_size} steps_per_epoch={steps_per_epoch:,} "
        f"total_steps={total_steps:,} warmup_steps={warmup_steps:,}",
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
        wandb_run.log(
            {
                "data/cache_open_seconds": cache_open_seconds,
                "data/cache_shards": len(dataset.shards),
                "data/cache_samples": len(dataset),
                "schedule/steps_per_epoch": steps_per_epoch,
                "schedule/total_steps": total_steps,
                "schedule/warmup_steps": warmup_steps,
            }
        )

    global_step = 0
    skipped_updates = 0
    total_training_start = time.perf_counter()
    try:
        for epoch in range(train_cfg.epochs):
            epoch_start = time.perf_counter()
            model.train()
            total_loss = 0.0
            correct = 0
            count = 0
            batches = dataset.iter_batches(
                batch_size=train_cfg.batch_size,
                shuffle_mode=train_cfg.shuffle_mode,
                seed=train_cfg.seed + epoch,
                max_samples=train_cfg.max_samples,
            )
            for batch in batches:
                result = train_batch(
                    batch=batch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    precision=precision,
                    grad_clip_norm=train_cfg.grad_clip_norm,
                )
                batch_count = len(batch)
                total_loss += result.loss * batch_count
                correct += result.correct
                count += batch_count
                if result.optimizer_stepped:
                    global_step += 1
                else:
                    skipped_updates += 1

                if (
                    result.optimizer_stepped
                    and global_step % train_cfg.log_every_steps == 0
                ):
                    elapsed = max(time.perf_counter() - epoch_start, 1e-9)
                    metrics = {
                        "train/batch_loss": result.loss,
                        "train/batch_accuracy": result.correct / batch_count,
                        "train/running_loss": total_loss / count,
                        "train/running_accuracy": correct / count,
                        "train/grad_norm": result.grad_norm,
                        "train/grad_scale": result.grad_scale,
                        "train/learning_rate": result.learning_rate,
                        "train/samples": count,
                        "train/samples_per_second": count / elapsed,
                        "train/epoch": epoch + 1,
                        "train/optimizer_step": global_step,
                        "train/skipped_optimizer_steps": skipped_updates,
                    }
                    print(
                        f"epoch={epoch + 1} samples={count:,} "
                        f"loss={metrics['train/running_loss']:.4f} "
                        f"acc={metrics['train/running_accuracy']:.3f} "
                        f"lr={result.learning_rate:.3e} "
                        f"scale={result.grad_scale:g} "
                        f"samples/s={metrics['train/samples_per_second']:.1f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(metrics)

            epoch_seconds = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch + 1,
                "samples": count,
                "batch_size": train_cfg.batch_size,
                "loss": total_loss / max(count, 1),
                "accuracy": correct / max(count, 1),
                "seconds": epoch_seconds,
                "samples_per_second": count / max(epoch_seconds, 1e-9),
                "optimizer_step": global_step,
                "skipped_optimizer_steps": skipped_updates,
            }
            history.append(row)
            print(row, flush=True)
            epoch_metrics = {
                "epoch/loss": row["loss"],
                "epoch/accuracy": row["accuracy"],
                "epoch/samples": row["samples"],
                "epoch/seconds": row["seconds"],
                "epoch/samples_per_second": row["samples_per_second"],
                "epoch/index": epoch + 1,
                "epoch/optimizer_step": global_step,
                "epoch/learning_rate": optimizer.param_groups[0]["lr"],
                "epoch/skipped_optimizer_steps": skipped_updates,
            }
            if device.type == "cuda":
                epoch_metrics["system/gpu_peak_memory_gb"] = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                )
                torch.cuda.reset_peak_memory_stats(device)
            if wandb_run is not None:
                wandb_run.log(epoch_metrics)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.to_dict(),
                    "experiment": {
                        "version_name": settings.version_name,
                        "train": asdict(train_cfg),
                        "model": asdict(model_cfg),
                        "wandb": asdict(wandb_cfg),
                        "cache_signature": feature_signature(config),
                    },
                    "history": history,
                },
                output_path / f"epoch-{epoch + 1}.pt",
            )
    finally:
        dataset.close()

    (output_path / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.summary["training/total_seconds"] = (
            time.perf_counter() - total_training_start
        )
        wandb_run.summary["training/final_loss"] = history[-1]["loss"]
        wandb_run.summary["training/final_accuracy"] = history[-1]["accuracy"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
