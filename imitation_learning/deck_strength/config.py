"""Strict configuration and deterministic unordered Deck-pair scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "cfg" / "deck_strength.yaml"


@dataclass(frozen=True, slots=True)
class DeckSpec:
    name: str
    cards: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    workers: int
    torch_threads_per_worker: int


@dataclass(frozen=True, slots=True)
class DeckStrengthSettings:
    cg_path: Path
    checkpoint: Path
    device: str
    seed: int
    games_per_pair: int
    output: Path
    runtime: RuntimeSettings
    decks: tuple[DeckSpec, ...]


@dataclass(frozen=True, slots=True)
class GameSpec:
    game_id: int
    deck_a: DeckSpec
    deck_b: DeckSpec
    deck_a_player: int
    seed: int


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{name}.{key} is required")
    return mapping[key]


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return value


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_settings(path: Path | None = None) -> DeckStrengthSettings:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        root = _mapping(yaml.safe_load(handle) or {}, "config")
    raw = _mapping(_required(root, "deck_strength", "config"), "deck_strength")
    runtime_raw = _mapping(
        _required(raw, "runtime", "deck_strength"), "deck_strength.runtime"
    )

    raw_decks = _required(raw, "decks", "deck_strength")
    if not isinstance(raw_decks, list) or len(raw_decks) < 2:
        raise ValueError("deck_strength.decks must contain at least two Decks")
    decks: list[DeckSpec] = []
    names: set[str] = set()
    for index, value in enumerate(raw_decks):
        item_name = f"deck_strength.decks[{index}]"
        deck_raw = _mapping(value, item_name)
        name = _required(deck_raw, "name", item_name)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{item_name}.name must be non-empty")
        name = name.strip()
        if name in names:
            raise ValueError("deck_strength deck names must be unique")
        cards = _required(deck_raw, "cards", item_name)
        if (
            not isinstance(cards, list)
            or len(cards) != 60
            or any(type(card) is not int or card < 0 for card in cards)
        ):
            raise ValueError(
                f"{item_name}.cards must contain exactly 60 non-negative "
                "integer Card IDs"
            )
        names.add(name)
        decks.append(DeckSpec(name=name, cards=tuple(cards)))

    device = _required(raw, "device", "deck_strength")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("deck_strength.device must be a non-empty string")
    seed = _required(raw, "seed", "deck_strength")
    if type(seed) is not int:
        raise ValueError("deck_strength.seed must be an integer")

    return DeckStrengthSettings(
        cg_path=_path(_required(raw, "cg_path", "deck_strength"), "cg_path"),
        checkpoint=_path(
            _required(raw, "checkpoint", "deck_strength"), "checkpoint"
        ),
        device=device.strip(),
        seed=seed,
        games_per_pair=_positive_int(
            _required(raw, "games_per_pair", "deck_strength"),
            "deck_strength.games_per_pair",
        ),
        output=_path(_required(raw, "output", "deck_strength"), "output"),
        runtime=RuntimeSettings(
            workers=_positive_int(
                _required(runtime_raw, "workers", "deck_strength.runtime"),
                "deck_strength.runtime.workers",
            ),
            torch_threads_per_worker=_positive_int(
                _required(
                    runtime_raw,
                    "torch_threads_per_worker",
                    "deck_strength.runtime",
                ),
                "deck_strength.runtime.torch_threads_per_worker",
            ),
        ),
        decks=tuple(decks),
    )


def schedule_games(settings: DeckStrengthSettings) -> tuple[GameSpec, ...]:
    games: list[GameSpec] = []
    game_id = 0
    for deck_a, deck_b in combinations(settings.decks, 2):
        for pair_game_index in range(settings.games_per_pair):
            games.append(
                GameSpec(
                    game_id=game_id,
                    deck_a=deck_a,
                    deck_b=deck_b,
                    deck_a_player=pair_game_index % 2,
                    seed=settings.seed + game_id,
                )
            )
            game_id += 1
    return tuple(games)
