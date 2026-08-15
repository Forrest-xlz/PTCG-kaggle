"""Strict YAML configuration and deterministic matchup scheduling."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "cfg" / "beam_search.yaml"


@dataclass(frozen=True, slots=True)
class DeckSpec:
    name: str
    cards: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SearchSettings:
    beam_width: int
    expansion_top_k: int
    alpha: float
    max_depth: int


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    workers: int
    torch_threads_per_worker: int


@dataclass(frozen=True, slots=True)
class BeamSearchSettings:
    cg_path: Path
    checkpoint: Path
    device: str
    seed: int
    games_per_matchup: int
    output: Path
    runtime: RuntimeSettings
    search: SearchSettings
    decks: tuple[DeckSpec, ...]


@dataclass(frozen=True, slots=True)
class GameSpec:
    game_id: int
    beam_deck: DeckSpec
    greedy_deck: DeckSpec
    beam_player: int
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


def load_settings(path: Path | None = None) -> BeamSearchSettings:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        root = _mapping(yaml.safe_load(handle) or {}, "config")
    raw = _mapping(_required(root, "beam_search", "config"), "beam_search")
    search_raw = _mapping(
        _required(raw, "search", "beam_search"), "beam_search.search"
    )
    runtime_raw = _mapping(
        _required(raw, "runtime", "beam_search"), "beam_search.runtime"
    )
    alpha_raw = _required(search_raw, "alpha", "beam_search.search")
    if type(alpha_raw) not in (int, float):
        raise ValueError("beam_search.search.alpha must be a finite number >= 0")
    alpha = float(alpha_raw)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("beam_search.search.alpha must be a finite number >= 0")

    raw_decks = _required(raw, "decks", "beam_search")
    if not isinstance(raw_decks, list) or not raw_decks:
        raise ValueError("beam_search.decks must be a non-empty list")
    decks: list[DeckSpec] = []
    names: set[str] = set()
    for index, value in enumerate(raw_decks):
        deck_raw = _mapping(value, f"beam_search.decks[{index}]")
        name = _required(deck_raw, "name", f"beam_search.decks[{index}]")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"beam_search.decks[{index}].name must be non-empty")
        name = name.strip()
        if name in names:
            raise ValueError("beam_search deck names must be unique")
        cards = _required(deck_raw, "cards", f"beam_search.decks[{index}]")
        if (
            not isinstance(cards, list)
            or len(cards) != 60
            or any(type(card) is not int or card < 0 for card in cards)
        ):
            raise ValueError(
                f"beam_search.decks[{index}].cards must contain exactly 60 "
                "non-negative integer Card IDs"
            )
        names.add(name)
        decks.append(DeckSpec(name=name, cards=tuple(cards)))

    device = _required(raw, "device", "beam_search")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("beam_search.device must be a non-empty string")
    seed = _required(raw, "seed", "beam_search")
    if type(seed) is not int:
        raise ValueError("beam_search.seed must be an integer")

    return BeamSearchSettings(
        cg_path=_path(_required(raw, "cg_path", "beam_search"), "cg_path"),
        checkpoint=_path(
            _required(raw, "checkpoint", "beam_search"), "checkpoint"
        ),
        device=device.strip(),
        seed=seed,
        games_per_matchup=_positive_int(
            _required(raw, "games_per_matchup", "beam_search"),
            "beam_search.games_per_matchup",
        ),
        output=_path(_required(raw, "output", "beam_search"), "output"),
        runtime=RuntimeSettings(
            workers=_positive_int(
                _required(runtime_raw, "workers", "beam_search.runtime"),
                "beam_search.runtime.workers",
            ),
            torch_threads_per_worker=_positive_int(
                _required(
                    runtime_raw,
                    "torch_threads_per_worker",
                    "beam_search.runtime",
                ),
                "beam_search.runtime.torch_threads_per_worker",
            ),
        ),
        search=SearchSettings(
            beam_width=_positive_int(
                _required(search_raw, "beam_width", "beam_search.search"),
                "beam_search.search.beam_width",
            ),
            expansion_top_k=_positive_int(
                _required(
                    search_raw, "expansion_top_k", "beam_search.search"
                ),
                "beam_search.search.expansion_top_k",
            ),
            alpha=alpha,
            max_depth=_positive_int(
                _required(search_raw, "max_depth", "beam_search.search"),
                "beam_search.search.max_depth",
            ),
        ),
        decks=tuple(decks),
    )


def schedule_games(settings: BeamSearchSettings) -> tuple[GameSpec, ...]:
    games: list[GameSpec] = []
    game_id = 0
    for beam_deck in settings.decks:
        for greedy_deck in settings.decks:
            for cell_index in range(settings.games_per_matchup):
                games.append(
                    GameSpec(
                        game_id=game_id,
                        beam_deck=beam_deck,
                        greedy_deck=greedy_deck,
                        beam_player=cell_index % 2,
                        seed=settings.seed + game_id,
                    )
                )
                game_id += 1
    return tuple(games)
