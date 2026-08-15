"""YAML-driven training for the encoder-only scalar value model."""
from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "cfg" / "value_train.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.attack_features import ATTACK_FEATURE_DIM
from model.card_features import CARD_FEATURE_DIM
from model.network import ModelConfig
from training.feature_cache import (
    ATTACK_DYNAMIC_DIM,
    CACHE_SCHEMA_VERSION,
    ENCODER_WORDS,
    HISTORY_STEPS,
    HISTORY_STRUCTURAL_DIM,
    MAX_ACTIONS,
    OPTION_CATEGORICAL_DIM,
    OPTION_NUMERIC_DIM,
    POKEMON_DYNAMIC_DIM,
    IndexBatch,
    MmapFeatureDataset,
    stable_deck_key,
)
from training.precision import PrecisionContext
from value.data import build_value_splits
from value.metrics import (
    RegressionAccumulator,
    evaluate_value_dataset,
    forward_encoder_batch,
)
from value.model import PTCGEncoderBackbone, PTCGValueModel


@dataclass(frozen=True)
class IsolationSettings:
    deck_data: str
    selections: dict[str, str]


@dataclass(frozen=True)
class ValueTrainSettings:
    cg_path: str
    data: str
    replay_episodes: str
    pretrained_checkpoint: str
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
    eval_every_steps: int
    save_every_steps: int
    save_every_epoch: bool
    validation_ratio: float
    validation_seed: int
    expert_validation_ratio: float
    seed: int
    device: str
    precision: str
    log_every_steps: int
    grad_clip_norm: float
    ema_alpha: float
    isolation_validation: IsolationSettings
    top_decks: list[list[int]]


@dataclass(frozen=True)
class ValueModelSettings:
    head_layers: int
    dropout: float
    output_activation: str


@dataclass(frozen=True)
class WandbSettings:
    enabled: bool
    project: str
    group: str
    name: str
    mode: str


@dataclass(frozen=True)
class ValueExperimentSettings:
    version_name: str
    train: ValueTrainSettings
    model: ValueModelSettings
    wandb: WandbSettings


@dataclass(frozen=True, slots=True)
class ValueBatchResult:
    mse: float
    rmse: float
    explained_variance: float
    samples: int
    grad_norm: float
    learning_rate: float
    optimizer_stepped: bool
    grad_scale: float


class ExponentialMovingAverage:
    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha < 1.0:
            raise ValueError("ema_alpha must be in [0, 1)")
        self.alpha = float(alpha)
        self.value: float | None = None

    def update(self, value: float) -> float:
        value = float(value)
        self.value = (
            value
            if self.value is None
            else self.alpha * self.value + (1.0 - self.alpha) * value
        )
        return self.value

    def state_dict(self) -> dict[str, float | None]:
        return {"alpha": self.alpha, "value": self.value}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not math.isclose(float(state["alpha"]), self.alpha):
            raise ValueError("EMA alpha does not match checkpoint")
        value = state.get("value")
        self.value = None if value is None else float(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _expand(value: Any, version_name: str) -> Any:
    if isinstance(value, str):
        return value.replace("${version_name}", version_name)
    if isinstance(value, list):
        return [_expand(item, version_name) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand(item, version_name) for key, item in value.items()}
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return value


def load_settings(path: Path | None = None) -> ValueExperimentSettings:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    raw = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")) or {},
        "config",
    )
    version_name = raw.get("version_name")
    if not isinstance(version_name, str) or not version_name.strip():
        raise ValueError("version_name must be a non-empty string")
    raw = _expand(raw, version_name.strip())
    train_raw = dict(_mapping(raw.get("value_train"), "value_train"))
    model_raw = dict(_mapping(raw.get("value_model"), "value_model"))
    wandb_raw = dict(_mapping(raw.get("wandb"), "wandb"))
    isolation_raw = dict(
        _mapping(
            train_raw.pop("isolation_validation", None),
            "value_train.isolation_validation",
        )
    )
    train_raw["isolation_validation"] = IsolationSettings(
        deck_data=str(isolation_raw.get("deck_data", "")),
        selections=dict(
            _mapping(
                isolation_raw.get("selections"),
                "value_train.isolation_validation.selections",
            )
        ),
    )
    try:
        train = ValueTrainSettings(**train_raw)
        model = ValueModelSettings(**model_raw)
        wandb = WandbSettings(**wandb_raw)
    except TypeError as exc:
        raise ValueError(f"invalid value training configuration: {exc}") from exc

    for name in (
        "epochs",
        "batch_size",
        "eval_every_steps",
        "save_every_steps",
        "log_every_steps",
    ):
        _positive_int(getattr(train, name), f"value_train.{name}")
    if type(model.head_layers) is not int or model.head_layers < 1:
        raise ValueError("value_model.head_layers must be an integer >= 1")
    if not 0.0 <= float(model.dropout) < 1.0:
        raise ValueError("value_model.dropout must be in [0, 1)")
    if model.output_activation != "tanh":
        raise ValueError("value_model.output_activation must be tanh")
    if not train.pretrained_checkpoint:
        raise ValueError("value_train.pretrained_checkpoint is required")
    if not 0.0 < float(train.validation_ratio) < 1.0:
        raise ValueError("value_train.validation_ratio must be in (0, 1)")
    if not 0.0 < float(train.expert_validation_ratio) <= 1.0:
        raise ValueError("expert_validation_ratio must be in (0, 1]")
    if not train.top_decks:
        raise ValueError("value_train.top_decks must not be empty")
    for index, deck in enumerate(train.top_decks):
        if len(deck) != 60 or any(type(card) is not int or card < 0 for card in deck):
            raise ValueError(f"value_train.top_decks[{index}] must contain 60 Card IDs")
    return ValueExperimentSettings(version_name.strip(), train, model, wandb)


def load_value_settings(path: Path | None = None) -> ValueExperimentSettings:
    """Load the standalone value-training configuration."""
    return load_settings(path)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device selected but CUDA is unavailable")
    return device


def should_trigger(interval: int, global_step: int) -> bool:
    return global_step > 0 and global_step % interval == 0


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
):
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    decay_steps = total_steps - warmup_steps

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        if decay_steps == 1:
            return 0.0
        progress = (step - warmup_steps) / (decay_steps - 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_value_batch(
    *,
    batch,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    precision: PrecisionContext,
    grad_clip_norm: float,
) -> ValueBatchResult:
    optimizer.zero_grad(set_to_none=True)
    learning_rate = float(optimizer.param_groups[0]["lr"])
    with precision.autocast():
        predictions, targets = forward_encoder_batch(batch, model, device)
        loss = torch.nn.functional.mse_loss(predictions, targets)
    metrics = RegressionAccumulator()
    metrics.update(predictions, targets)
    finalized = metrics.finalize()
    step = precision.backward_step(
        loss, model, optimizer, scheduler, grad_clip_norm
    )
    return ValueBatchResult(
        mse=float(loss.item()),
        rmse=finalized.rmse,
        explained_variance=finalized.explained_variance,
        samples=finalized.samples,
        grad_norm=step.grad_norm,
        learning_rate=learning_rate,
        optimizer_stepped=step.optimizer_stepped,
        grad_scale=step.grad_scale,
    )


def value_checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    precision,
    encoder_config: Mapping[str, Any],
    value_model_config: Mapping[str, Any],
    global_step: int,
    epoch: int,
    ema: Mapping[str, Any],
    history: list[dict],
    experiment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": precision.state_dict(),
        "encoder_config": dict(encoder_config),
        "value_model_config": dict(value_model_config),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "ema": dict(ema),
        "history": list(history),
        "experiment": dict(experiment or {}),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }


def validation_namespaces(
    top_deck_count: int, isolation_names: tuple[str, ...]
) -> tuple[str, ...]:
    names = [
        "val_in_distribution",
        "val_in_distribution_expert",
        "val_latest",
        "val_latest_expert",
        *(
            name if name.startswith("val_") else f"val_{name}"
            for name in isolation_names
        ),
    ]
    for index in range(1, top_deck_count + 1):
        names.extend(
            (
                f"val_in_distribution_deck{index}",
                f"val_in_distribution_expert_deck{index}",
                f"val_latest_deck{index}",
                f"val_latest_expert_deck{index}",
            )
        )
    return tuple(names)


def feature_signature(config: ModelConfig) -> dict[str, Any]:
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


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    return checkpoint


def _model_from_checkpoint(settings: ValueExperimentSettings):
    train = settings.train
    resume_checkpoint = None
    if train.resume:
        if not train.resume_checkpoint:
            raise ValueError("resume_checkpoint is required when resume=true")
        resume_checkpoint = _load_checkpoint(project_path(train.resume_checkpoint))
        config_raw = resume_checkpoint.get("encoder_config")
        value_state = resume_checkpoint.get("model")
        if not isinstance(value_state, Mapping):
            raise ValueError("value checkpoint is missing model weights")
        policy_state = {
            name.removeprefix("backbone."): tensor
            for name, tensor in value_state.items()
            if name.startswith("backbone.")
        }
    else:
        policy_checkpoint = _load_checkpoint(
            project_path(train.pretrained_checkpoint)
        )
        config_raw = policy_checkpoint.get("config")
        policy_state = policy_checkpoint.get("model")
    if not isinstance(config_raw, Mapping) or not isinstance(policy_state, Mapping):
        raise ValueError("checkpoint must contain model weights and architecture")
    config = ModelConfig(**dict(config_raw))
    backbone = PTCGEncoderBackbone.from_policy_state(
        config,
        torch.zeros((config.card_count, CARD_FEATURE_DIM)),
        torch.zeros((config.attack_count, ATTACK_FEATURE_DIM)),
        policy_state,
    )
    model = PTCGValueModel(
        backbone,
        head_layers=settings.model.head_layers,
        dropout=settings.model.dropout,
    )
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
    return model, config, resume_checkpoint


def _metric_payload(namespace: str, metrics) -> dict[str, float | int]:
    return {
        f"{namespace}/rmse": metrics.rmse,
        f"{namespace}/explained_variance": metrics.explained_variance,
        f"{namespace}/samples": metrics.samples,
        f"{namespace}/target_mean": metrics.target_mean,
        f"{namespace}/prediction_mean": metrics.prediction_mean,
    }


def main() -> None:
    settings = load_settings()
    train = settings.train
    random.seed(train.seed)
    np.random.seed(train.seed)
    torch.manual_seed(train.seed)
    device = resolve_device(train.device)
    precision = PrecisionContext(train.precision, device)
    model, encoder_config, resume_checkpoint = _model_from_checkpoint(settings)
    model.to(device)
    dataset = MmapFeatureDataset(project_path(train.data), feature_signature(encoder_config))
    try:
        from training.expert_validation import load_expert_date_info
        from training.isolation_validation import load_isolation_replay_sets

        expert_info = load_expert_date_info(
            project_path(train.replay_episodes),
            dataset.shard_dates,
            train.expert_validation_ratio,
        )
        isolation = load_isolation_replay_sets(
            project_path(train.isolation_validation.deck_data),
            {
                name: project_path(path)
                for name, path in train.isolation_validation.selections.items()
            },
            dataset.shard_dates,
        )
        splits = build_value_splits(
            dataset,
            validation_ratio=train.validation_ratio,
            validation_seed=train.validation_seed,
            expert_episode_keys={
                date: info.expert_episode_keys for date, info in expert_info.items()
            },
            top_deck_keys=tuple(stable_deck_key(deck) for deck in train.top_decks),
            isolation_episode_keys=isolation.by_namespace,
        )
        steps_per_epoch = math.ceil(len(splits.train) / train.batch_size)
        total_steps = steps_per_epoch * train.epochs
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train.learning_rate,
            weight_decay=train.weight_decay,
            betas=(train.beta1, train.beta2),
        )
        scheduler = build_lr_scheduler(
            optimizer, total_steps=total_steps, warmup_steps=train.warmup_steps
        )
        ema = {
            "mse": ExponentialMovingAverage(train.ema_alpha),
            "rmse": ExponentialMovingAverage(train.ema_alpha),
        }
        history: list[dict] = []
        global_step = 0
        start_epoch = 0
        if resume_checkpoint is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
            precision.load_state_dict(resume_checkpoint.get("scaler", {}))
            global_step = int(resume_checkpoint["global_step"])
            start_epoch = int(resume_checkpoint["epoch"])
            history = list(resume_checkpoint.get("history", []))
            for name, tracker in ema.items():
                tracker.load_state_dict(resume_checkpoint["ema"][name])

        wandb_run = None
        if settings.wandb.enabled:
            import wandb

            wandb_run = wandb.init(
                project=settings.wandb.project,
                group=settings.wandb.group or None,
                name=settings.wandb.name or None,
                mode=settings.wandb.mode,
                config={
                    "version_name": settings.version_name,
                    "value_train": asdict(train),
                    "value_model": asdict(settings.model),
                    "encoder": encoder_config.to_dict(),
                },
            )
            wandb.define_metric("optimizer_step")
            wandb.define_metric("train/*", step_metric="optimizer_step")
            for namespace in validation_namespaces(
                len(train.top_decks), tuple(sorted(splits.isolation_masks))
            ):
                wandb.define_metric(f"{namespace}/*", step_metric="optimizer_step")

        output = project_path(train.output)
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "version_name": settings.version_name,
                    "value_train": asdict(train),
                    "value_model": asdict(settings.model),
                    "wandb": asdict(settings.wandb),
                    "encoder": encoder_config.to_dict(),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        def save(name: str, epoch: int) -> None:
            torch.save(
                value_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    precision=precision,
                    encoder_config=encoder_config.to_dict(),
                    value_model_config=asdict(settings.model),
                    global_step=global_step,
                    epoch=epoch,
                    ema={key: tracker.state_dict() for key, tracker in ema.items()},
                    history=history,
                    experiment={"version_name": settings.version_name},
                ),
                checkpoint_dir / name,
            )

        def log_validation() -> None:
            evaluations = (
                (
                    "val_latest",
                    splits.latest,
                    splits.latest_masks,
                ),
                (
                    "val_in_distribution",
                    splits.in_distribution,
                    splits.in_distribution_masks,
                ),
                (
                    "isolation_union",
                    splits.isolation,
                    splits.isolation_masks,
                ),
            )
            for namespace, indices, masks in evaluations:
                result = evaluate_value_dataset(
                    model=model,
                    dataset=dataset,
                    indices=indices,
                    batch_size=train.batch_size,
                    device=device,
                    precision=precision,
                    subgroup_masks=masks,
                )
                payload = (
                    {}
                    if namespace == "isolation_union"
                    else _metric_payload(namespace, result.overall)
                )
                for subgroup, metrics in result.subgroups.items():
                    logged_subgroup = (
                        subgroup
                        if subgroup.startswith("val_")
                        else f"val_{subgroup}"
                    )
                    payload.update(_metric_payload(logged_subgroup, metrics))
                payload["optimizer_step"] = global_step
                if wandb_run is not None:
                    wandb_run.log(payload)
                print(
                    f"{namespace} step={global_step:,} "
                    f"rmse={result.overall.rmse:.5f} "
                    f"explained_variance={result.overall.explained_variance:.5f} "
                    f"samples={result.overall.samples:,}",
                    flush=True,
                )

        print(
            f"value_version={settings.version_name} device={device} "
            f"precision={train.precision} train={len(splits.train):,} "
            f"val_in_distribution={len(splits.in_distribution):,} "
            f"val_latest={len(splits.latest):,} "
            f"val_isolation={len(splits.isolation):,} "
            f"steps_per_epoch={steps_per_epoch:,} total_steps={total_steps:,}",
            flush=True,
        )
        last_eval = -1
        for epoch_index in range(start_epoch, train.epochs):
            epoch = epoch_index + 1
            started = time.perf_counter()
            data_started = started
            data_wait_seconds = 0.0
            compute_seconds = 0.0
            samples = 0
            for index_batch in dataset.iter_index_batches(
                splits.train,
                train.batch_size,
                seed=train.seed + epoch_index,
                shuffle=True,
            ):
                batch = dataset.collate_encoder(index_batch)
                compute_started = time.perf_counter()
                data_wait_seconds += compute_started - data_started
                result = train_value_batch(
                    batch=batch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    precision=precision,
                    grad_clip_norm=train.grad_clip_norm,
                )
                compute_seconds += time.perf_counter() - compute_started
                samples += result.samples
                if result.optimizer_stepped:
                    global_step += 1
                mse_ema = ema["mse"].update(result.mse)
                rmse_ema = ema["rmse"].update(result.rmse)
                if result.optimizer_stepped and should_trigger(
                    train.log_every_steps, global_step
                ):
                    measured_seconds = data_wait_seconds + compute_seconds
                    data_wait_percent = (
                        100.0 * data_wait_seconds / measured_seconds
                        if measured_seconds > 0.0
                        else 0.0
                    )
                    samples_per_second = (
                        samples / measured_seconds
                        if measured_seconds > 0.0
                        else 0.0
                    )
                    payload = {
                        "train/mse_ema": mse_ema,
                        "train/rmse_ema": rmse_ema,
                        "train/explained_variance": result.explained_variance,
                        "train/learning_rate": result.learning_rate,
                        "train/grad_norm": result.grad_norm,
                        "train/grad_scale": result.grad_scale,
                        "train/data_wait_percent": data_wait_percent,
                        "train/samples_per_second": samples_per_second,
                        "optimizer_step": global_step,
                    }
                    if wandb_run is not None:
                        wandb_run.log(payload)
                    print(
                        f"epoch={epoch} step={global_step:,} samples={samples:,} "
                        f"mse_ema={mse_ema:.5f} rmse_ema={rmse_ema:.5f} "
                        f"lr={result.learning_rate:.3e} "
                        f"data_wait={data_wait_percent:.1f}% "
                        f"samples/s={samples_per_second:.1f}",
                        flush=True,
                    )
                if (
                    result.optimizer_stepped
                    and should_trigger(train.eval_every_steps, global_step)
                    and last_eval != global_step
                ):
                    log_validation()
                    last_eval = global_step
                    model.train()
                if result.optimizer_stepped and should_trigger(
                    train.save_every_steps, global_step
                ):
                    save(f"step-{global_step:07d}.pt", epoch_index)
                data_started = time.perf_counter()
            epoch_record = {
                "epoch": epoch,
                "optimizer_step": global_step,
                "samples": samples,
                "seconds": time.perf_counter() - started,
                "data_wait_seconds": data_wait_seconds,
                "compute_seconds": compute_seconds,
            }
            history.append(epoch_record)
            if train.save_every_epoch:
                save(f"epoch-{epoch:03d}.pt", epoch)
        if last_eval != global_step:
            log_validation()
        if wandb_run is not None:
            wandb_run.finish()
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
