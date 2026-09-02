"""Evaluate one policy checkpoint on the training-defined validation splits."""
from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.attack_features import build_attack_feature_table
from model.card_features import build_card_feature_table
from model.features import (
    ATTACK_DYNAMIC_DIM,
    HISTORY_STEPS,
    HISTORY_STRUCTURAL_DIM,
    OPTION_CATEGORICAL_DIM,
    OPTION_NUMERIC_DIM,
    POKEMON_DYNAMIC_DIM,
)
from model.network import ModelConfig, PTCGTransformer
from training.expert_validation import load_expert_date_info
from training.feature_cache import (
    CACHE_SCHEMA_VERSION,
    ENCODER_WORDS,
    MAX_ACTIONS,
    MmapFeatureDataset,
    stable_deck_key,
)
from training.isolation_validation import load_isolation_replay_sets
from training.precision import PrecisionContext
from validation.config import ValidationSettings, load_settings
from validation.metrics import (
    evaluate_dataset,
    format_metrics,
    top_deck_subgroup_masks,
)


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    namespace: str
    indices: np.ndarray
    subgroup_masks: dict[str, np.ndarray]


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    for key in ("model", "config"):
        if key not in checkpoint:
            raise ValueError(f"checkpoint is missing {key!r}")
        if not isinstance(checkpoint[key], Mapping):
            raise ValueError(f"checkpoint {key!r} must be a mapping")
    return checkpoint


def model_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> ModelConfig:
    raw = checkpoint.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint 'config' must be a mapping")
    try:
        return ModelConfig(**dict(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint model config: {exc}") from exc


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


def validate_compatible_configs(
    configs: Sequence[ModelConfig],
) -> dict[str, Any]:
    configs = tuple(configs)
    if not configs:
        raise ValueError("at least one checkpoint config is required")
    reference = feature_signature(configs[0])
    for checkpoint_index, config in enumerate(configs[1:], start=2):
        signature = feature_signature(config)
        differences = [
            name
            for name, value in reference.items()
            if signature.get(name) != value
        ]
        if differences:
            raise ValueError(
                f"checkpoint {checkpoint_index} has an incompatible cache "
                f"signature: {', '.join(differences)}"
            )
    return reference


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device selected but CUDA is unavailable")
    return device


def evaluation_specs(splits) -> tuple[EvaluationSpec, ...]:
    latest_masks = {
        "val_latest_expert": splits.latest_expert_mask,
        **top_deck_subgroup_masks(
            "val_latest",
            splits.latest_top_deck_masks,
            splits.latest_expert_top_deck_masks,
        ),
    }
    in_distribution_masks = {
        "val_in_distribution_expert": splits.in_distribution_expert_mask,
        **top_deck_subgroup_masks(
            "val_in_distribution",
            splits.in_distribution_top_deck_masks,
            splits.in_distribution_expert_top_deck_masks,
        ),
    }
    specs = [
        EvaluationSpec("val_latest", splits.latest, latest_masks),
        EvaluationSpec(
            "val_in_distribution",
            splits.in_distribution,
            in_distribution_masks,
        ),
    ]
    if len(splits.isolation):
        specs.insert(
            0,
            EvaluationSpec(
                "isolation_union", splits.isolation, splits.isolation_masks
            ),
        )
    return tuple(specs)


def _configure_engine(settings: ValidationSettings) -> None:
    engine_root = settings.cg_path
    if engine_root.name == "cg":
        engine_root = engine_root.parent
    if not (engine_root / "cg" / "__init__.py").is_file():
        raise FileNotFoundError(
            "train.cg_path must contain the cg package; "
            f"not found under: {engine_root}"
        )
    if str(engine_root) not in sys.path:
        sys.path.insert(0, str(engine_root))


def _build_model(
    checkpoint: Mapping[str, Any],
    config: ModelConfig,
) -> PTCGTransformer:
    from cg.api import all_attack, all_card_data

    model = PTCGTransformer(
        config,
        build_card_feature_table(all_card_data(), config.card_count),
        build_attack_feature_table(all_attack(), config.attack_count),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def _build_splits(
    dataset: MmapFeatureDataset,
    settings: ValidationSettings,
):
    selection_paths = {
        f"val_{name}": path
        for name, path in settings.isolation_selections.items()
    }
    isolation_sets = (
        load_isolation_replay_sets(
            deck_data_dir=settings.isolation_deck_data,
            selection_paths=selection_paths,
            required_dates=dataset.shard_dates,
        )
        if selection_paths
        else None
    )
    expert_dates = load_expert_date_info(
        replay_root=settings.replay_episodes,
        required_dates=dataset.shard_dates,
        ratio=settings.expert_validation_ratio,
    )
    top_deck_keys = tuple(stable_deck_key(deck) for deck in settings.top_decks)
    if len(set(top_deck_keys)) != len(top_deck_keys):
        raise ValueError("train.top_decks contains duplicate exact decks")
    splits = dataset.build_splits(
        validation_ratio=settings.validation_ratio,
        validation_seed=settings.validation_seed,
        expert_episode_keys={
            date: info.expert_episode_keys
            for date, info in expert_dates.items()
        },
        top_deck_keys=top_deck_keys,
        train_replay_ratio=1.0,
        train_replay_seed=0,
        isolation_episode_keys=(
            None if isolation_sets is None else isolation_sets.by_namespace
        ),
    )
    return splits, isolation_sets, expert_dates


def _print_split_context(splits, isolation_sets, expert_dates) -> None:
    for date in sorted(expert_dates):
        info = expert_dates[date]
        print(
            f"expert_date={date[0]}.{date[1]} cutoff={info.cutoff:g} "
            f"episodes={info.episode_count:,} "
            f"expert_episodes={info.expert_episode_count:,}",
            flush=True,
        )
    if isolation_sets is not None:
        for namespace in sorted(isolation_sets.by_namespace):
            print(
                f"{namespace} selected_decks="
                f"{isolation_sets.selected_deck_counts[namespace]:,} "
                f"replays={isolation_sets.replay_counts[namespace]:,} "
                f"samples={splits.isolation_sample_counts[namespace]:,}",
                flush=True,
            )
    print(
        f"validation_splits isolation_union={len(splits.isolation):,} "
        f"in_distribution={len(splits.in_distribution):,} "
        f"latest={len(splits.latest):,} "
        f"latest_date={splits.latest_date[0]}.{splits.latest_date[1]}",
        flush=True,
    )


def main() -> None:
    settings = load_settings()
    _configure_engine(settings)
    device = resolve_device(settings.device)
    precision = PrecisionContext(settings.precision, device)
    models: list[PTCGTransformer] = []
    configs: list[ModelConfig] = []
    for checkpoint_index, checkpoint_path in enumerate(
        settings.checkpoints, start=1
    ):
        checkpoint = load_checkpoint(checkpoint_path)
        config = model_config_from_checkpoint(checkpoint)
        validate_compatible_configs((*configs, config))
        model = _build_model(checkpoint, config).to(device)
        model.eval()
        models.append(model)
        configs.append(config)
        print(
            f"checkpoint_{checkpoint_index}={checkpoint_path} "
            f"d_model={config.d_model} heads={config.num_heads} "
            f"encoder_layers={config.encoder_layers} "
            f"decoder_layers={config.decoder_layers} "
            f"norm={config.norm_mode}",
            flush=True,
        )
        del checkpoint

    shared_signature = validate_compatible_configs(configs)
    config = configs[0]
    invalid_card_ids = sorted(
        card_id
        for deck in settings.top_decks
        for card_id in deck
        if card_id >= config.card_count
    )
    if invalid_card_ids:
        raise ValueError(
            "train.top_decks contains card IDs outside the model vocabulary: "
            f"{invalid_card_ids}"
        )

    print(
        f"ensemble={settings.ensemble_enabled} models={len(models)} "
        f"device={device} "
        f"precision={settings.precision} batch_size={settings.batch_size:,} "
        f"averaging={'probabilities' if settings.ensemble_enabled else 'off'}",
        flush=True,
    )

    dataset = MmapFeatureDataset(
        settings.data,
        expected_signature=shared_signature,
    )
    try:
        splits, isolation_sets, expert_dates = _build_splits(dataset, settings)
        _print_split_context(splits, isolation_sets, expert_dates)
        for spec in evaluation_specs(splits):
            result = evaluate_dataset(
                models=models,
                ensemble_enabled=settings.ensemble_enabled,
                dataset=dataset,
                indices=spec.indices,
                batch_size=settings.batch_size,
                device=device,
                precision=precision,
                subgroup_masks=spec.subgroup_masks,
            )
            if spec.namespace == "isolation_union":
                print(
                    f"isolation_union samples={len(spec.indices):,} "
                    f"seconds={result.seconds:.3f}",
                    flush=True,
                )
            else:
                print(
                    format_metrics(
                        spec.namespace, result.overall, result.seconds
                    ),
                    flush=True,
                )
            for namespace, metrics in result.subgroups.items():
                print(format_metrics(namespace, metrics), flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
