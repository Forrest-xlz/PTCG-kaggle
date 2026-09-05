"""Configuration boundary for the standalone validation evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "cfg" / "validate_policy.yaml"
ISOLATION_SELECTION_NAMES = (
    "deck_isolation",
    "archetype_isolation",
    "top_deck_archetype_isolation",
)
PRECISIONS = frozenset({"fp32", "fp16", "bf16"})


@dataclass(frozen=True, slots=True)
class ValidationSettings:
    version_name: str
    train_config: Path
    ensemble_enabled: bool
    checkpoints: tuple[Path, ...]
    cg_path: Path
    data: Path
    replay_episodes: Path
    batch_size: int
    device: str
    precision: str
    validation_ratio: float
    validation_seed: int
    expert_validation_ratio: float
    isolation_deck_data: Path
    isolation_selections: dict[str, Path]
    top_decks: tuple[tuple[int, ...], ...]
    top_deck_names: tuple[str, ...] = ()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, scope: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{scope}.{key} is required")
    return mapping[key]


def _interpolate(value: Any, version_name: str) -> Any:
    if isinstance(value, str):
        return value.replace("${version_name}", version_name)
    if isinstance(value, list):
        return [_interpolate(item, version_name) for item in value]
    if isinstance(value, dict):
        return {
            key: _interpolate(item, version_name)
            for key, item in value.items()
        }
    return value


def _project_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _ratio(value: Any, name: str, *, allow_one: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    upper_valid = result <= 1 if allow_one else result < 1
    if result <= 0 or not upper_valid:
        interval = "(0, 1]" if allow_one else "(0, 1)"
        raise ValueError(f"{name} must be in {interval}")
    return result


def _load_yaml(path: Path, scope: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{scope} config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return _mapping(yaml.safe_load(handle) or {}, scope)


def load_settings(path: Path = CONFIG_PATH) -> ValidationSettings:
    validation_path = Path(path).resolve()
    raw_validation = _mapping(
        _required(_load_yaml(validation_path, "validation"), "validation", "root"),
        "validation",
    )
    train_config = _project_path(
        _required(raw_validation, "train_config", "validation"),
        "validation.train_config",
    )
    raw_train_root = _load_yaml(train_config, "train")
    version_name = _required(raw_train_root, "version_name", "root")
    if not isinstance(version_name, str) or not version_name.strip():
        raise ValueError("version_name must be a non-empty string")
    train = _mapping(
        _interpolate(_required(raw_train_root, "train", "root"), version_name),
        "train",
    )
    validation = _mapping(
        _interpolate(raw_validation, version_name), "validation"
    )

    batch_size = _positive_int(
        _required(validation, "batch_size", "validation"),
        "validation.batch_size",
    )
    device = _required(validation, "device", "validation")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("validation.device must be a non-empty string")
    precision = _required(validation, "precision", "validation")
    if precision not in PRECISIONS:
        raise ValueError(
            f"validation.precision must be one of {sorted(PRECISIONS)}"
        )

    ensemble = _mapping(
        _required(validation, "ensemble", "validation"),
        "validation.ensemble",
    )
    ensemble_enabled = _required(
        ensemble, "enabled", "validation.ensemble"
    )
    if type(ensemble_enabled) is not bool:
        raise ValueError("validation.ensemble.enabled must be true or false")
    raw_checkpoints = _required(
        ensemble, "checkpoints", "validation.ensemble"
    )
    if not isinstance(raw_checkpoints, list):
        raise ValueError("validation.ensemble.checkpoints must be a list")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in raw_checkpoints
    ):
        raise ValueError(
            "validation.ensemble.checkpoints must contain non-empty paths"
        )
    checkpoints = tuple(
        _project_path(value, "validation.ensemble.checkpoints")
        for value in raw_checkpoints
    )
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError(
            "validation.ensemble.checkpoints must contain distinct paths"
        )
    if ensemble_enabled and len(checkpoints) < 2:
        raise ValueError(
            "enabled validation ensemble requires at least two checkpoints"
        )
    if not ensemble_enabled and len(checkpoints) != 1:
        raise ValueError(
            "disabled validation ensemble requires exactly one checkpoint"
        )

    validation_seed = _required(train, "validation_seed", "train")
    if isinstance(validation_seed, bool) or not isinstance(validation_seed, int):
        raise ValueError("train.validation_seed must be an integer")

    isolation = _mapping(
        _required(train, "isolation_validation", "train"),
        "train.isolation_validation",
    )
    raw_selections = _mapping(
        _required(isolation, "selections", "train.isolation_validation"),
        "train.isolation_validation.selections",
    )
    if set(raw_selections) != set(ISOLATION_SELECTION_NAMES):
        raise ValueError(
            "train.isolation_validation.selections must contain exactly "
            f"{list(ISOLATION_SELECTION_NAMES)}"
        )
    selections = {
        name: _project_path(
            raw_selections[name],
            f"train.isolation_validation.selections.{name}",
        )
        for name in ISOLATION_SELECTION_NAMES
        if raw_selections[name] is not None
    }

    from validation.deck_names import parse_top_decks
    top_decks, top_deck_names = parse_top_decks(_required(train, "top_decks", "train"))

    return ValidationSettings(
        version_name=version_name,
        train_config=train_config,
        ensemble_enabled=ensemble_enabled,
        checkpoints=checkpoints,
        cg_path=_project_path(
            _required(train, "cg_path", "train"), "train.cg_path"
        ),
        data=_project_path(_required(train, "data", "train"), "train.data"),
        replay_episodes=_project_path(
            _required(train, "replay_episodes", "train"),
            "train.replay_episodes",
        ),
        batch_size=batch_size,
        device=device,
        precision=precision,
        validation_ratio=_ratio(
            _required(train, "validation_ratio", "train"),
            "train.validation_ratio",
            allow_one=False,
        ),
        validation_seed=validation_seed,
        expert_validation_ratio=_ratio(
            _required(train, "expert_validation_ratio", "train"),
            "train.expert_validation_ratio",
            allow_one=True,
        ),
        isolation_deck_data=_project_path(
            _required(isolation, "deck_data", "train.isolation_validation"),
            "train.isolation_validation.deck_data",
        ),
        isolation_selections=selections,
        top_decks=tuple(tuple(deck) for deck in top_decks),
        top_deck_names=top_deck_names,
    )
