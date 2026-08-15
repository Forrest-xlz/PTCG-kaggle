"""Checkpoint-backed policy inference shared by Greedy and Beam agents."""
from __future__ import annotations

import sys
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from beam_search.config import BeamSearchSettings, DeckSpec
from model.attack_features import build_attack_feature_table
from model.card_features import build_card_feature_table
from model.features import (
    ATTACK_DYNAMIC_DIM,
    HISTORY_STEPS,
    HISTORY_STRUCTURAL_DIM,
    OPTION_CATEGORICAL_DIM,
    POKEMON_DYNAMIC_DIM,
    HistoryActionFeatures,
    NumericFeatureCatalog,
    OptionFeatures,
    _default_numeric_catalog,
    decoder_features,
    encoder_features,
    enumerate_actions,
    history_action_features,
)
from model.network import ModelConfig, PTCGTransformer


@dataclass(slots=True)
class PolicyHistory:
    actions: deque[HistoryActionFeatures] = field(
        default_factory=lambda: deque(maxlen=HISTORY_STEPS)
    )

    def append(self, action: HistoryActionFeatures) -> None:
        self.actions.append(action)

    def clone(self) -> "PolicyHistory":
        return PolicyHistory(deque(self.actions, maxlen=HISTORY_STEPS))

    def clear(self) -> None:
        self.actions.clear()


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    actions: tuple[list[int], ...]
    probabilities: torch.Tensor
    log_probabilities: torch.Tensor
    encoded_options: OptionFeatures | None = None


def normalize_policy(
    actions: Sequence[Sequence[int]],
    logits: torch.Tensor,
    encoded_options: OptionFeatures | None = None,
) -> PolicyOutput:
    flat_logits = logits.float().reshape(-1)
    if flat_logits.numel() != len(actions):
        raise ValueError(
            "policy action count does not match logits: "
            f"{len(actions)} actions, {flat_logits.numel()} logits"
        )
    probabilities = torch.softmax(flat_logits, dim=0)
    log_probabilities = probabilities.clamp_min(
        torch.finfo(torch.float32).tiny
    ).log()
    return PolicyOutput(
        actions=tuple([int(index) for index in action] for action in actions),
        probabilities=probabilities,
        log_probabilities=log_probabilities,
        encoded_options=encoded_options,
    )


def _rows_or_empty(
    parts: list[np.ndarray], width: int, dtype: np.dtype
) -> np.ndarray:
    return (
        np.concatenate(parts, axis=0)
        if parts
        else np.empty((0, width), dtype=dtype)
    )


def history_tensors(
    history: PolicyHistory,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    recent = list(history.actions)[-HISTORY_STEPS:]
    slots: list[HistoryActionFeatures | None] = (
        [None] * (HISTORY_STEPS - len(recent)) + recent
    )
    select_type = np.zeros((1, HISTORY_STEPS), dtype=np.int64)
    select_context = np.zeros((1, HISTORY_STEPS), dtype=np.int64)
    valid = np.zeros((1, HISTORY_STEPS), dtype=np.bool_)
    categorical: list[np.ndarray] = []
    structural: list[np.ndarray] = []
    pokemon: list[np.ndarray] = []
    attack: list[np.ndarray] = []
    offsets = [0]
    for slot_index, action in enumerate(slots):
        if action is not None:
            select_type[0, slot_index] = action.select_type
            select_context[0, slot_index] = action.select_context
            valid[0, slot_index] = True
            categorical.append(action.option_categorical)
            structural.append(action.structural)
            pokemon.append(action.pokemon_dynamic)
            attack.append(action.attack_dynamic)
            offsets.append(offsets[-1] + len(action.option_categorical))
        else:
            offsets.append(offsets[-1])
    return (
        torch.as_tensor(select_type, device=device),
        torch.as_tensor(select_context, device=device),
        torch.as_tensor(valid, device=device),
        torch.as_tensor(
            _rows_or_empty(categorical, OPTION_CATEGORICAL_DIM, np.int64),
            device=device,
        ),
        torch.as_tensor(
            _rows_or_empty(structural, HISTORY_STRUCTURAL_DIM, np.int64),
            device=device,
        ),
        torch.as_tensor(
            _rows_or_empty(pokemon, POKEMON_DYNAMIC_DIM, np.float32),
            device=device,
        ),
        torch.as_tensor(
            _rows_or_empty(attack, ATTACK_DYNAMIC_DIM, np.float32),
            device=device,
        ),
        torch.as_tensor(offsets, dtype=torch.long, device=device),
    )


def _sparse_tensors(vector: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(vector.index, dtype=torch.long, device=device),
        torch.as_tensor(vector.value, dtype=torch.float32, device=device),
        torch.as_tensor(vector.offset, dtype=torch.long, device=device),
    )


def _dense(values: Sequence[float], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device).unsqueeze(0)


class PolicyAgent:
    def __init__(
        self,
        model: PTCGTransformer,
        config: ModelConfig,
        catalog: NumericFeatureCatalog,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.catalog = catalog
        self.device = device

    def evaluate(
        self,
        obs: Any,
        deck: DeckSpec,
        history: PolicyHistory,
    ) -> PolicyOutput:
        actions = enumerate_actions(
            len(obs.select.option),
            int(obs.select.minCount),
            int(obs.select.maxCount),
            limit=64,
        )
        if not actions:
            raise ValueError("observation produced no legal actions")
        encoder = encoder_features(
            obs,
            list(deck.cards),
            self.config.card_count,
            numeric_catalog=self.catalog,
        )
        options = decoder_features(
            obs,
            actions,
            self.config.card_count,
            self.config.attack_count,
            numeric_catalog=self.catalog,
        )
        inputs = (
            *_sparse_tensors(encoder.sparse, self.device),
            torch.as_tensor(
                encoder.pokemon_appear, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            _dense(encoder.own_summary, self.device),
            _dense(encoder.opponent_summary, self.device),
            _dense(encoder.global_summary, self.device),
            *history_tensors(history, self.device),
            torch.as_tensor(
                options.categorical, dtype=torch.long, device=self.device
            ),
            torch.as_tensor(
                options.numeric, dtype=torch.float32, device=self.device
            ),
            torch.as_tensor(
                options.pokemon_dynamic,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                options.attack_dynamic,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                options.action_index, dtype=torch.long, device=self.device
            ),
            torch.as_tensor(
                options.action_offset, dtype=torch.long, device=self.device
            ),
        )
        with torch.inference_mode():
            logits = self.model(*inputs)
        return normalize_policy(actions, logits, options)

    def record_action(
        self,
        obs: Any,
        selected: list[int],
        history: PolicyHistory,
        encoded_options: OptionFeatures | None = None,
    ) -> None:
        history.append(
            history_action_features(
                obs,
                selected,
                self.config.card_count,
                self.config.attack_count,
                numeric_catalog=self.catalog,
                encoded_options=encoded_options,
            )
        )

    def greedy(
        self,
        obs: Any,
        deck: DeckSpec,
        history: PolicyHistory,
        *,
        record: bool = False,
    ) -> list[int]:
        output = self.evaluate(obs, deck, history)
        selected = output.actions[int(output.probabilities.argmax().item())]
        if record:
            self.record_action(obs, selected, history, output.encoded_options)
        return selected


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("checkpoint must contain a model mapping")
    if not isinstance(checkpoint.get("config"), Mapping):
        raise ValueError("checkpoint must contain a config mapping")
    return checkpoint


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device selected but CUDA is unavailable")
    return device


def _configure_engine(path: Path) -> None:
    root = path.parent if path.name == "cg" else path
    if not (root / "cg" / "__init__.py").is_file():
        raise FileNotFoundError(f"cg package not found under: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_policy_agent(settings: BeamSearchSettings) -> PolicyAgent:
    _configure_engine(settings.cg_path)
    from cg.api import all_attack, all_card_data

    checkpoint = _load_checkpoint(settings.checkpoint)
    try:
        config = ModelConfig(**dict(checkpoint["config"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint model config: {exc}") from exc
    cards = all_card_data()
    attacks = all_attack()
    model = PTCGTransformer(
        config,
        build_card_feature_table(cards, config.card_count),
        build_attack_feature_table(attacks, config.attack_count),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = _resolve_device(settings.device)
    model.to(device).eval()
    catalog = _default_numeric_catalog(config.card_count)
    return PolicyAgent(model, config, catalog, device)
