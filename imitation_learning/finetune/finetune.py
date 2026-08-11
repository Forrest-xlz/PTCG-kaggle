"""Fine-tune a policy on generic replays plus one expert exact deck."""
from __future__ import annotations

import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import AbstractSet, Mapping

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "cfg" / "finetune.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train import (
    ExponentialMovingAverage,
    ExperimentSettings,
    WandbSettings,
    _log_validation,
    build_lr_scheduler,
    evaluate_dataset,
    feature_signature,
    format_loser_augmentation_line,
    load_settings,
    resolve_device,
    select_loser_augmentation_dates,
    should_trigger,
    top_deck_subgroup_masks,
    train_batch,
)
from cg.api import all_attack, all_card_data
from model.attack_features import build_attack_feature_table
from model.card_features import build_card_feature_table
from model.network import ModelConfig, PTCGTransformer
from training.expert_validation import (
    load_expert_date_info,
    load_expert_loser_date_info,
)
from training.feature_cache import (
    MmapFeatureDataset,
    PLAYER_RESULT_WIN,
    stable_deck_key,
)
from training.isolation_validation import load_isolation_replay_sets
from training.precision import PrecisionContext


@dataclass(frozen=True)
class FineTuneSettings:
    version_name: str
    train_config: Path
    checkpoint: Path
    output: str
    epochs: int
    in_distribution_ratio: float
    in_distribution_seed: int
    expert_ratio: float
    deck: tuple[int, ...]
    learning_rate: float
    wandb: WandbSettings
    base: ExperimentSettings


@dataclass(frozen=True, slots=True)
class FineTuneData:
    indices: np.ndarray
    generic_indices: np.ndarray
    expert_deck_indices: np.ndarray
    overlap_samples: int


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"finetune.{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"finetune.{name} must be finite")
    return result


def load_finetune_settings(path: Path = CONFIG_PATH) -> FineTuneSettings:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tune config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("The fine-tune YAML root must be a mapping")
    required_root = {"version_name", "finetune", "wandb"}
    if set(raw) != required_root:
        raise ValueError(
            "Fine-tune YAML sections mismatch: "
            f"missing={sorted(required_root - set(raw))} "
            f"unknown={sorted(set(raw) - required_root)}"
        )
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
    values = raw["finetune"]
    if not isinstance(values, dict):
        raise ValueError("finetune must be a mapping")
    required = {
        "train_config",
        "checkpoint",
        "output",
        "epochs",
        "in_distribution_ratio",
        "in_distribution_seed",
        "expert_ratio",
        "deck",
        "learning_rate",
    }
    if set(values) != required:
        raise ValueError(
            "finetune fields mismatch: "
            f"missing={sorted(required - set(values))} "
            f"unknown={sorted(set(values) - required)}"
        )

    train_config = _project_path(values["train_config"])
    if not train_config.is_file():
        raise FileNotFoundError(f"Referenced train config not found: {train_config}")
    checkpoint = _project_path(values["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Fine-tune checkpoint not found: {checkpoint}")
    base = load_settings(train_config)

    epochs = values["epochs"]
    if type(epochs) is not int or epochs < 1:
        raise ValueError("finetune.epochs must be a positive integer")
    in_distribution_seed = values["in_distribution_seed"]
    if type(in_distribution_seed) is not int:
        raise ValueError("finetune.in_distribution_seed must be an integer")
    in_distribution_ratio = _finite_number(
        values["in_distribution_ratio"], "in_distribution_ratio"
    )
    if not 0 < in_distribution_ratio <= 1:
        raise ValueError("finetune.in_distribution_ratio must be in (0, 1]")
    expert_ratio = _finite_number(values["expert_ratio"], "expert_ratio")
    if not 0 < expert_ratio <= 1:
        raise ValueError("finetune.expert_ratio must be in (0, 1]")

    deck = values["deck"]
    if not isinstance(deck, list) or len(deck) != 60:
        raise ValueError("finetune.deck must contain exactly 60 card IDs")
    if any(type(card_id) is not int or card_id < 0 for card_id in deck):
        raise ValueError("finetune.deck card IDs must be non-negative integers")

    learning_rate = values["learning_rate"]
    if learning_rate is None:
        learning_rate = base.train.learning_rate
    else:
        learning_rate = _finite_number(learning_rate, "learning_rate")
        if learning_rate <= 0:
            raise ValueError("finetune.learning_rate must be > 0")

    wandb_raw = raw["wandb"]
    if not isinstance(wandb_raw, dict):
        raise ValueError("wandb must be a mapping")
    wandb = WandbSettings(**wandb_raw)
    if wandb.enabled and not wandb.project:
        raise ValueError("wandb.project is required when wandb.enabled is true")

    return FineTuneSettings(
        version_name=version_name,
        train_config=train_config,
        checkpoint=checkpoint,
        output=str(values["output"]),
        epochs=epochs,
        in_distribution_ratio=in_distribution_ratio,
        in_distribution_seed=in_distribution_seed,
        expert_ratio=expert_ratio,
        deck=tuple(deck),
        learning_rate=float(learning_rate),
        wandb=wandb,
        base=base,
    )


def _sorted_indices(value: np.ndarray, name: str) -> np.ndarray:
    indices = np.asarray(value)
    if (
        indices.ndim != 1
        or (indices.size > 1 and np.any(indices[1:] <= indices[:-1]))
    ):
        raise ValueError(f"{name} must be a sorted one-dimensional array")
    return indices


def build_finetune_indices(
    dataset,
    full_train_indices: np.ndarray,
    generic_train_indices: np.ndarray,
    expert_episode_keys: Mapping[tuple[int, int], AbstractSet[int]],
    deck_key: int,
) -> FineTuneData:
    full = _sorted_indices(full_train_indices, "full training indices")
    generic = _sorted_indices(generic_train_indices, "generic training indices")
    if generic.size == 0:
        raise ValueError("generic fine-tune component is empty")
    if not np.all(np.isin(generic, full, assume_unique=True)):
        raise ValueError("generic training indices must be contained in full training")
    missing_dates = set(dataset.shard_dates) - set(expert_episode_keys)
    if missing_dates:
        raise ValueError(f"expert episode keys are missing dates: {missing_dates}")

    parts: list[np.ndarray] = []
    for shard, start, end, date in zip(
        dataset.shards,
        dataset.starts,
        dataset.ends,
        dataset.shard_dates,
    ):
        left = int(np.searchsorted(full, start, side="left"))
        right = int(np.searchsorted(full, end, side="left"))
        shard_global = full[left:right]
        if shard_global.size == 0:
            continue
        local = shard_global - int(start)
        date_keys = np.fromiter(expert_episode_keys[date], dtype=np.uint32)
        mask = np.isin(
            shard.arrays["episode_key"][local],
            date_keys,
            assume_unique=False,
        )
        mask &= shard.arrays["deck_key"][local] == int(deck_key)
        mask &= shard.arrays["player_result"][local] == PLAYER_RESULT_WIN
        if np.any(mask):
            parts.append(shard_global[mask])

    if not parts:
        raise ValueError("expert-deck fine-tune component is empty")
    expert_deck = np.concatenate(parts).astype(full.dtype, copy=False)
    overlap = int(
        np.intersect1d(generic, expert_deck, assume_unique=True).size
    )
    combined = np.concatenate((generic, expert_deck)).astype(
        full.dtype, copy=False
    )
    return FineTuneData(
        indices=combined,
        generic_indices=generic,
        expert_deck_indices=expert_deck,
        overlap_samples=overlap,
    )


def epoch_finetune_indices(data: FineTuneData, seed: int) -> np.ndarray:
    indices = data.indices.copy()
    np.random.default_rng(seed).shuffle(indices)
    return indices


def load_finetune_checkpoint(path: Path) -> dict:
    path = Path(path)
    match = re.fullmatch(r"epoch-(\d+)\.pt", path.name)
    if match is None:
        raise ValueError("fine-tuning requires an epoch-NNN.pt checkpoint")
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tune checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Fine-tune checkpoint root must be a mapping")
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
            f"Fine-tuning requires a complete checkpoint; missing={missing}"
        )
    epoch = checkpoint["epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("checkpoint epoch must be a positive integer")
    if int(match.group(1)) != epoch:
        raise ValueError("checkpoint filename epoch does not match its payload")
    if not isinstance(checkpoint["config"], dict):
        raise ValueError("checkpoint config must be a mapping")
    return checkpoint


def restore_finetune_state(
    checkpoint: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    precision,
    learning_rate: float,
) -> None:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
        group["initial_lr"] = float(learning_rate)
    precision.load_state_dict(checkpoint["scaler"])


def create_finetune_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    sample_count: int,
    batch_size: int,
    warmup_steps: int,
):
    steps_per_epoch = math.ceil(sample_count / batch_size)
    total_steps = steps_per_epoch * epochs
    scheduler = build_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    return scheduler, steps_per_epoch, total_steps


def _same_array(left, right) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def assert_same_validation_splits(full, sampled) -> None:
    array_names = (
        "isolation",
        "in_distribution",
        "in_distribution_expert_mask",
        "latest",
        "latest_expert_mask",
    )
    if any(
        not _same_array(getattr(full, name), getattr(sampled, name))
        for name in array_names
    ):
        raise ValueError("full and sampled validation splits differ")
    if full.latest_date != sampled.latest_date:
        raise ValueError("full and sampled validation dates differ")
    if set(full.isolation_masks) != set(sampled.isolation_masks):
        raise ValueError("full and sampled validation masks differ")
    for name in full.isolation_masks:
        if not _same_array(
            full.isolation_masks[name], sampled.isolation_masks[name]
        ):
            raise ValueError("full and sampled validation masks differ")
    tuple_names = (
        "in_distribution_top_deck_masks",
        "in_distribution_expert_top_deck_masks",
        "latest_top_deck_masks",
        "latest_expert_top_deck_masks",
    )
    for name in tuple_names:
        left = getattr(full, name)
        right = getattr(sampled, name)
        if len(left) != len(right) or any(
            not _same_array(a, b) for a, b in zip(left, right)
        ):
            raise ValueError("full and sampled validation subgroup masks differ")


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    precision,
    config: ModelConfig,
    settings: FineTuneSettings,
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
            "kind": "finetune",
            "version_name": settings.version_name,
            "source_checkpoint": str(settings.checkpoint),
            "finetune": {
                "epochs": settings.epochs,
                "in_distribution_ratio": settings.in_distribution_ratio,
                "in_distribution_seed": settings.in_distribution_seed,
                "expert_ratio": settings.expert_ratio,
                "deck": list(settings.deck),
                "learning_rate": settings.learning_rate,
            },
            "train": asdict(settings.base.train),
            "wandb": asdict(settings.wandb),
        },
        "ema": {name: tracker.state_dict() for name, tracker in ema.items()},
        "history": history,
    }


def _validation_namespaces(deck_count: int) -> list[str]:
    namespaces = [
        "train/*",
        "epoch/*",
        "val_in_distribution/*",
        "val_in_distribution_expert/*",
        "val_latest/*",
        "val_latest_expert/*",
        "val_deck_isolation/*",
        "val_archetype_isolation/*",
        "val_top_deck_archetype_isolation/*",
    ]
    for deck_index in range(1, deck_count + 1):
        namespaces.extend(
            [
                f"val_in_distribution_deck{deck_index}/*",
                f"val_in_distribution_expert_deck{deck_index}/*",
                f"val_latest_deck{deck_index}/*",
                f"val_latest_expert_deck{deck_index}/*",
            ]
        )
    return namespaces


def main() -> None:
    settings = load_finetune_settings()
    base_train = settings.base.train
    random.seed(base_train.seed)
    np.random.seed(base_train.seed)
    torch.manual_seed(base_train.seed)

    checkpoint = load_finetune_checkpoint(settings.checkpoint)
    config = ModelConfig(**checkpoint["config"])
    device = resolve_device(base_train.device)
    precision = PrecisionContext(base_train.precision, device)
    data_path = _project_path(base_train.data)
    replay_root = _project_path(base_train.replay_episodes)
    dataset = MmapFeatureDataset(
        data_path,
        expected_signature=feature_signature(config),
    )
    wandb_run = None
    try:
        isolation_cfg = base_train.isolation_validation
        isolation_sets = load_isolation_replay_sets(
            deck_data_dir=_project_path(isolation_cfg.deck_data),
            selection_paths={
                f"val_{name}": _project_path(path)
                for name, path in isolation_cfg.selections.items()
            },
            required_dates=dataset.shard_dates,
        )
        validation_experts = load_expert_date_info(
            replay_root,
            required_dates=dataset.shard_dates,
            ratio=base_train.expert_validation_ratio,
        )
        validation_expert_keys = {
            date: info.expert_episode_keys
            for date, info in validation_experts.items()
        }
        top_deck_keys = tuple(
            stable_deck_key(deck) for deck in base_train.top_decks
        )

        if base_train.loser_augmentation.enabled:
            loser_dates = select_loser_augmentation_dates(
                dataset.shard_dates,
                base_train.loser_augmentation.recent_dates,
            )
            loser_info = load_expert_loser_date_info(
                replay_root,
                required_dates=loser_dates,
                ratio=base_train.loser_augmentation.expert_ratio,
            )
            loser_keys = {
                date: info.eligible_episode_keys
                for date, info in loser_info.items()
            }
        else:
            loser_dates = ()
            loser_info = {}
            loser_keys = None

        split_kwargs = {
            "validation_ratio": base_train.validation_ratio,
            "validation_seed": base_train.validation_seed,
            "expert_episode_keys": validation_expert_keys,
            "top_deck_keys": top_deck_keys,
            "isolation_episode_keys": isolation_sets.by_namespace,
            "loser_episode_keys": loser_keys,
        }
        full_splits = dataset.build_splits(
            **split_kwargs,
            train_replay_ratio=1.0,
            train_replay_seed=settings.in_distribution_seed,
        )
        generic_splits = dataset.build_splits(
            **split_kwargs,
            train_replay_ratio=settings.in_distribution_ratio,
            train_replay_seed=settings.in_distribution_seed,
        )
        assert_same_validation_splits(full_splits, generic_splits)

        finetune_experts = load_expert_date_info(
            replay_root,
            required_dates=dataset.shard_dates,
            ratio=settings.expert_ratio,
        )
        finetune_data = build_finetune_indices(
            dataset=dataset,
            full_train_indices=full_splits.train,
            generic_train_indices=generic_splits.train,
            expert_episode_keys={
                date: info.expert_episode_keys
                for date, info in finetune_experts.items()
            },
            deck_key=stable_deck_key(settings.deck),
        )

        sample_count = len(finetune_data.indices)
        if base_train.max_samples is not None:
            sample_count = min(sample_count, base_train.max_samples)
        cards = all_card_data()
        attacks = all_attack()
        model = PTCGTransformer(
            config,
            build_card_feature_table(cards, config.card_count),
            build_attack_feature_table(attacks, config.attack_count),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=base_train.weight_decay,
            betas=(base_train.beta1, base_train.beta2),
        )
        restore_finetune_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            precision=precision,
            learning_rate=settings.learning_rate,
        )
        scheduler, steps_per_epoch, total_steps = create_finetune_scheduler(
            optimizer,
            epochs=settings.epochs,
            sample_count=sample_count,
            batch_size=base_train.batch_size,
            warmup_steps=base_train.warmup_steps,
        )
        ema = {
            name: ExponentialMovingAverage(base_train.ema_alpha)
            for name in ("loss", "top1", "top3", "top5")
        }
        history: list[dict] = []
        global_step = 0

        if settings.wandb.enabled:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError(
                    "wandb.enabled is true but wandb is not installed"
                ) from exc
            wandb_run = wandb.init(
                project=settings.wandb.project,
                group=settings.wandb.group or None,
                name=settings.wandb.name or None,
                mode=settings.wandb.mode,
                config={
                    "version_name": settings.version_name,
                    "finetune": {
                        "checkpoint": str(settings.checkpoint),
                        "epochs": settings.epochs,
                        "in_distribution_ratio": settings.in_distribution_ratio,
                        "in_distribution_seed": settings.in_distribution_seed,
                        "expert_ratio": settings.expert_ratio,
                        "deck": list(settings.deck),
                        "learning_rate": settings.learning_rate,
                    },
                    "train": asdict(base_train),
                    "network": config.to_dict(),
                },
            )
            wandb.define_metric("optimizer_step")
            for namespace in _validation_namespaces(len(top_deck_keys)):
                wandb.define_metric(namespace, step_metric="optimizer_step")

        output_root = (
            Path(wandb_run.dir).parent / "local-output"
            if wandb_run is not None
            else _project_path(settings.output)
        )
        checkpoint_dir = output_root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resolved = {
            "version_name": settings.version_name,
            "finetune": {
                "train_config": str(settings.train_config),
                "checkpoint": str(settings.checkpoint),
                "output": settings.output,
                "epochs": settings.epochs,
                "in_distribution_ratio": settings.in_distribution_ratio,
                "in_distribution_seed": settings.in_distribution_seed,
                "expert_ratio": settings.expert_ratio,
                "deck": list(settings.deck),
                "learning_rate": settings.learning_rate,
            },
            "wandb": asdict(settings.wandb),
        }
        (output_root / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )

        for date, info in sorted(finetune_experts.items()):
            print(
                f"finetune_expert_date={date[0]}.{date[1]} "
                f"ratio={settings.expert_ratio:g} cutoff={info.cutoff:g} "
                f"episodes={info.episode_count:,} "
                f"expert_episodes={info.expert_episode_count:,}",
                flush=True,
            )
        for date in loser_dates:
            print(
                format_loser_augmentation_line(
                    loser_info[date], full_splits.loser_augmentation_counts[date]
                ),
                flush=True,
            )
        print(
            f"finetune_source={settings.checkpoint} "
            f"full_train_samples={len(full_splits.train):,} "
            f"generic_replays={generic_splits.selected_train_replays:,} "
            f"generic_samples={len(finetune_data.generic_indices):,} "
            f"expert_deck_samples={len(finetune_data.expert_deck_indices):,} "
            f"overlap_samples={finetune_data.overlap_samples:,} "
            f"samples_per_epoch={sample_count:,} "
            f"steps_per_epoch={steps_per_epoch:,} total_steps={total_steps:,}",
            flush=True,
        )
        if wandb_run is not None:
            data_metrics = {
                "data/full_train_samples": len(full_splits.train),
                "data/generic_train_replays": generic_splits.selected_train_replays,
                "data/generic_train_samples": len(finetune_data.generic_indices),
                "data/expert_deck_samples": len(finetune_data.expert_deck_indices),
                "data/overlap_samples": finetune_data.overlap_samples,
                "data/finetune_samples_per_epoch": sample_count,
                "schedule/steps_per_epoch": steps_per_epoch,
                "schedule/total_steps": total_steps,
                "schedule/warmup_steps": base_train.warmup_steps,
                "optimizer_step": 0,
            }
            for date, info in finetune_experts.items():
                prefix = f"data/finetune_expert_{date[0]}_{date[1]}"
                data_metrics.update(
                    {
                        f"{prefix}_ratio": settings.expert_ratio,
                        f"{prefix}_cutoff": info.cutoff,
                        f"{prefix}_episodes": info.episode_count,
                        f"{prefix}_expert_episodes": info.expert_episode_count,
                    }
                )
            wandb_run.log(data_metrics)

        def save_checkpoint(name: str, epoch: int) -> None:
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    precision,
                    config,
                    settings,
                    global_step,
                    epoch,
                    ema,
                    history,
                ),
                checkpoint_dir / name,
            )

        last_eval_step = -1

        def run_validation() -> None:
            nonlocal last_eval_step
            isolation_result = evaluate_dataset(
                model=model,
                dataset=dataset,
                indices=full_splits.isolation,
                batch_size=base_train.batch_size,
                device=device,
                precision=precision,
                subgroup_masks=full_splits.isolation_masks,
            )
            for namespace, metrics in sorted(
                isolation_result.subgroups.items()
            ):
                _log_validation(
                    namespace, metrics, None, global_step, wandb_run
                )
            for namespace, indices, masks in (
                (
                    "val_latest",
                    full_splits.latest,
                    {
                        "val_latest_expert": full_splits.latest_expert_mask,
                        **top_deck_subgroup_masks(
                            "val_latest",
                            full_splits.latest_top_deck_masks,
                            full_splits.latest_expert_top_deck_masks,
                        ),
                    },
                ),
                (
                    "val_in_distribution",
                    full_splits.in_distribution,
                    {
                        "val_in_distribution_expert": (
                            full_splits.in_distribution_expert_mask
                        ),
                        **top_deck_subgroup_masks(
                            "val_in_distribution",
                            full_splits.in_distribution_top_deck_masks,
                            full_splits.in_distribution_expert_top_deck_masks,
                        ),
                    },
                ),
            ):
                result = evaluate_dataset(
                    model=model,
                    dataset=dataset,
                    indices=indices,
                    batch_size=base_train.batch_size,
                    device=device,
                    precision=precision,
                    subgroup_masks=masks,
                )
                _log_validation(
                    namespace,
                    result.overall,
                    result.seconds,
                    global_step,
                    wandb_run,
                )
                for subgroup, metrics in result.subgroups.items():
                    _log_validation(
                        subgroup, metrics, None, global_step, wandb_run
                    )
            last_eval_step = global_step

        train_samples_seen = 0
        train_seconds = 0.0
        skipped_updates = 0
        for epoch_index in range(settings.epochs):
            epoch = epoch_index + 1
            epoch_started = time.perf_counter()
            epoch_loss = 0.0
            epoch_top1 = epoch_top3 = epoch_top5 = epoch_samples = 0
            model.train()
            epoch_indices = epoch_finetune_indices(
                finetune_data, seed=base_train.seed + epoch_index
            )
            for batch in dataset.iter_batches(
                indices=epoch_indices,
                batch_size=base_train.batch_size,
                seed=base_train.seed + epoch_index,
                shuffle=False,
                max_samples=base_train.max_samples,
            ):
                started = time.perf_counter()
                result = train_batch(
                    batch,
                    model,
                    optimizer,
                    scheduler,
                    device,
                    precision,
                    base_train.grad_clip_norm,
                )
                train_seconds += time.perf_counter() - started
                metrics = result.metrics
                averages = metrics.averages()
                ema_values = {
                    "loss": ema["loss"].update(averages["loss"]),
                    "top1": ema["top1"].update(averages["top1_accuracy"]),
                    "top3": ema["top3"].update(averages["top3_accuracy"]),
                    "top5": ema["top5"].update(averages["top5_accuracy"]),
                }
                train_samples_seen += metrics.samples
                epoch_loss += metrics.loss * metrics.samples
                epoch_top1 += metrics.top1_correct
                epoch_top3 += metrics.top3_correct
                epoch_top5 += metrics.top5_correct
                epoch_samples += metrics.samples
                if not result.optimizer_stepped:
                    skipped_updates += 1
                    continue
                global_step += 1
                if should_trigger(base_train.log_every_steps, global_step):
                    payload = {
                        "train/ema_loss": ema_values["loss"],
                        "train/ema_top1_accuracy": ema_values["top1"],
                        "train/ema_top3_accuracy": ema_values["top3"],
                        "train/ema_top5_accuracy": ema_values["top5"],
                        "train/learning_rate": result.learning_rate,
                        "train/grad_norm": result.grad_norm,
                        "train/samples_per_second": train_samples_seen / max(train_seconds, 1e-9),
                        "optimizer_step": global_step,
                    }
                    print(
                        f"epoch={epoch} step={global_step:,} "
                        f"samples={train_samples_seen:,} "
                        f"ema_loss={ema_values['loss']:.4f} "
                        f"ema_top1={ema_values['top1']:.3f} "
                        f"lr={result.learning_rate:.3e}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(payload)
                if should_trigger(base_train.eval_every_steps, global_step):
                    run_validation()
                if should_trigger(base_train.save_every_steps, global_step):
                    save_checkpoint(f"step-{global_step:08d}.pt", epoch)

            epoch_metrics = {
                "epoch": epoch,
                "samples": epoch_samples,
                "loss": epoch_loss / epoch_samples,
                "top1_accuracy": epoch_top1 / epoch_samples,
                "top3_accuracy": epoch_top3 / epoch_samples,
                "top5_accuracy": epoch_top5 / epoch_samples,
                "seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_metrics)
            print(epoch_metrics, flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"epoch/{key}": value for key, value in epoch_metrics.items() if key != "epoch"},
                        "epoch/index": epoch,
                        "optimizer_step": global_step,
                    }
                )
            if last_eval_step != global_step:
                run_validation()
            if base_train.save_every_epoch:
                save_checkpoint(f"epoch-{epoch:03d}.pt", epoch)

        print(
            f"fine-tuning complete optimizer_steps={global_step:,} "
            f"skipped_optimizer_steps={skipped_updates:,}",
            flush=True,
        )
    finally:
        dataset.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
