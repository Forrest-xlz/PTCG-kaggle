"""Shared model, cache, and tensor helpers for expert classification."""
from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from expert_classifier.config import PROJECT_ROOT, ExpertClassifierSettings, project_path
from model.attack_features import build_attack_feature_table
from model.card_features import build_card_feature_table
from model.network import ModelConfig, PTCGTransformer
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
    CachedBatch,
)


def bootstrap_cg(train_raw: Mapping[str, Any]) -> None:
    value = train_raw["train"].get("cg_path")
    if not value:
        raise ValueError("referenced train.train.cg_path is required")
    path = project_path(str(value)).resolve()
    if path.name == "cg":
        path = path.parent
    if not (path / "cg" / "__init__.py").exists():
        raise FileNotFoundError(f"cg package not found under {path}")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_torch_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    return payload


def model_config_from_train(
    train_raw: Mapping[str, Any], card_count: int, attack_count: int
) -> ModelConfig:
    values = dict(train_raw["model"])
    multiplier = values.pop("ffn_multiplier")
    values["d_feedforward"] = int(values["d_model"] * multiplier)
    values.update(card_count=card_count, attack_count=attack_count)
    return ModelConfig(**values)


def model_config_from_checkpoint(payload: Mapping[str, Any]) -> ModelConfig:
    raw = payload.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    return ModelConfig(**dict(raw))


def build_backbone(config: ModelConfig, cards, attacks) -> PTCGTransformer:
    return PTCGTransformer(
        config,
        build_card_feature_table(cards, config.card_count),
        build_attack_feature_table(attacks, config.attack_count),
    )


def load_bc_weights(backbone: PTCGTransformer, payload: Mapping[str, Any]) -> None:
    raw = payload.get("model")
    if not isinstance(raw, Mapping):
        raise ValueError("BC checkpoint model must be a mapping")
    state = dict(raw)
    if any(str(key).startswith("backbone.") for key in state):
        state = {
            str(key)[len("backbone."):]: value
            for key, value in state.items()
            if str(key).startswith("backbone.")
        }
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    allowed_missing = {"decoder_fc.weight", "decoder_fc.bias"}
    bad_missing = set(missing) - allowed_missing
    if bad_missing or unexpected:
        raise ValueError(
            f"incompatible BC backbone: missing={sorted(bad_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


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


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device selected but CUDA is unavailable")
    return device


def _tensor(array, device: torch.device, dtype=None) -> torch.Tensor:
    return torch.from_numpy(array).to(
        device=device,
        dtype=dtype,
        non_blocking=device.type == "cuda",
    )


def forward_cached_batch(model, batch: CachedBatch, device: torch.device) -> torch.Tensor:
    return model(
        _tensor(batch.encoder_index, device),
        _tensor(batch.encoder_value, device, torch.float32),
        _tensor(batch.encoder_offset, device),
        _tensor(batch.encoder_pokemon_appear, device, torch.long),
        _tensor(batch.own_summary, device, torch.float32),
        _tensor(batch.opponent_summary, device, torch.float32),
        _tensor(batch.global_summary, device, torch.float32),
        _tensor(batch.history_select_type, device, torch.long),
        _tensor(batch.history_select_context, device, torch.long),
        _tensor(batch.history_valid, device, torch.long),
        _tensor(batch.history_option_categorical, device, torch.long),
        _tensor(batch.history_structural, device, torch.long),
        _tensor(batch.history_pokemon_dynamic, device, torch.float32),
        _tensor(batch.history_attack_dynamic, device, torch.float32),
        _tensor(batch.history_option_offset, device, torch.long),
        _tensor(batch.option_categorical, device, torch.long),
        _tensor(batch.option_numeric, device, torch.float32),
        _tensor(batch.pokemon_dynamic, device, torch.float32),
        _tensor(batch.attack_dynamic, device, torch.float32),
        _tensor(batch.action_option_index, device, torch.long),
        _tensor(batch.action_option_offset, device, torch.long),
        target=_tensor(batch.target, device, torch.long),
    )


def build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int):
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    decay_steps = total_steps - warmup_steps

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if decay_steps == 1:
            return 0.0
        progress = min(1.0, max(0.0, (step - warmup_steps) / (decay_steps - 1)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
