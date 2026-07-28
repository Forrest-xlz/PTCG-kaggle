"""Resolve reviewed exact-deck selections into replay-level validation sets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from training.feature_cache import (
    parse_source_date,
    stable_deck_key,
    stable_episode_key,
)


DateKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class IsolationReplaySets:
    by_namespace: dict[str, dict[DateKey, frozenset[int]]]
    selected_deck_counts: dict[str, int]
    replay_counts: dict[str, int]
    union_replay_count: int
    pairwise_overlap_counts: dict[str, int]


def _parse_deck(value: object, source: str) -> list[int]:
    if not isinstance(value, str):
        raise ValueError(f"{source} deck must be JSON text")
    try:
        deck = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} contains malformed deck JSON") from exc
    if not isinstance(deck, list):
        raise ValueError(f"{source} deck must decode to a list")
    try:
        stable_deck_key(deck)
    except ValueError as exc:
        raise ValueError(f"{source} contains an invalid deck: {exc}") from exc
    return deck


def _load_selected_decks(
    selection_paths: Mapping[str, Path],
) -> tuple[dict[str, set[int]], dict[int, set[str]]]:
    selected: dict[str, set[int]] = {}
    labels: dict[int, set[str]] = {}
    for namespace, path_value in selection_paths.items():
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(
                f"{namespace} selection CSV not found: {path}"
            )
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "card_ids" not in frame.columns:
            raise ValueError(f"{path} is missing the card_ids column")
        if frame.empty:
            raise ValueError(f"{path} contains no selected decks")
        keys: set[int] = set()
        for row_number, row in enumerate(
            frame.to_dict("records"),
            start=2,
        ):
            source = f"{path}:{row_number} card_ids"
            deck = _parse_deck(row["card_ids"], source)
            key = stable_deck_key(deck)
            keys.add(key)
            label = str(row.get("deck_id", key))
            labels.setdefault(key, set()).add(label)
        selected[str(namespace)] = keys
    if not selected:
        raise ValueError("at least one isolation selection is required")
    return selected, labels


def load_isolation_replay_sets(
    deck_data_dir: Path,
    selection_paths: Mapping[str, Path],
    required_dates: Iterable[DateKey],
) -> IsolationReplaySets:
    """Match either player's deck and return per-date episode-key sets."""
    deck_root = Path(deck_data_dir)
    if not deck_root.is_dir():
        raise FileNotFoundError(
            f"isolation deck-data directory not found: {deck_root}"
        )
    deck_paths = sorted(deck_root.glob("*.decks.csv"))
    if not deck_paths:
        raise FileNotFoundError(
            f"no *.decks.csv files found under {deck_root}"
        )
    dates = set(required_dates)
    if not dates:
        raise ValueError("required_dates must not be empty")

    selected, labels = _load_selected_decks(selection_paths)
    namespaces_by_deck: dict[int, set[str]] = {}
    for namespace, keys in selected.items():
        for key in keys:
            namespaces_by_deck.setdefault(key, set()).add(namespace)

    matched_decks: set[int] = set()
    replay_sets: dict[str, dict[DateKey, set[int]]] = {
        namespace: {date: set() for date in dates}
        for namespace in selected
    }
    key_cache: dict[str, int] = {}
    for path in deck_paths:
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={"date": str, "episode_id": str, "deck": str},
            usecols=lambda column: column in {"date", "episode_id", "deck"},
        )
        required_columns = {"date", "episode_id", "deck"}
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )
        for row_number, row in enumerate(
            frame.itertuples(index=False),
            start=2,
        ):
            deck_text = str(row.deck)
            deck_key = key_cache.get(deck_text)
            if deck_key is None:
                deck_key = stable_deck_key(
                    _parse_deck(deck_text, f"{path}:{row_number} deck")
                )
                key_cache[deck_text] = deck_key
            namespaces = namespaces_by_deck.get(deck_key)
            if not namespaces:
                continue
            matched_decks.add(deck_key)
            date = parse_source_date(str(row.date))
            if date not in dates:
                continue
            episode_key = stable_episode_key(row.episode_id)
            for namespace in namespaces:
                replay_sets[namespace][date].add(episode_key)

    missing_decks = sorted(set(namespaces_by_deck) - matched_decks)
    if missing_decks:
        missing_labels = sorted(
            label
            for key in missing_decks
            for label in labels.get(key, {str(key)})
        )
        raise ValueError(
            "selected exact decks are absent from deck extracts: "
            f"{missing_labels}"
        )

    frozen = {
        namespace: {
            date: frozenset(keys)
            for date, keys in sorted(date_sets.items())
        }
        for namespace, date_sets in sorted(replay_sets.items())
    }
    identities = {
        namespace: {
            (date, episode_key)
            for date, keys in date_sets.items()
            for episode_key in keys
        }
        for namespace, date_sets in frozen.items()
    }
    empty = sorted(
        namespace
        for namespace, replay_ids in identities.items()
        if not replay_ids
    )
    if empty:
        raise ValueError(
            "isolation selections resolve to no required-date replays: "
            f"{empty}"
        )
    union = set().union(*identities.values())
    overlap_counts = {
        f"{first}__{second}": len(
            identities[first] & identities[second]
        )
        for first, second in combinations(sorted(identities), 2)
    }
    return IsolationReplaySets(
        by_namespace=frozen,
        selected_deck_counts={
            namespace: len(keys)
            for namespace, keys in sorted(selected.items())
        },
        replay_counts={
            namespace: len(replay_ids)
            for namespace, replay_ids in sorted(identities.items())
        },
        union_replay_count=len(union),
        pairwise_overlap_counts=overlap_counts,
    )
