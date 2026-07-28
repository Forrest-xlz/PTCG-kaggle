"""Reusable exact-deck census, archetype, and similarity analysis."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


ARCHETYPE_RULES: tuple[dict[str, Any], ...] = (
    {"name": "Great Tusk / Crustle", "all": ["Great Tusk", "Crustle"]},
    {
        "name": "Marnie Grimmsnarl",
        "any": ["Marnie's Grimmsnarl ex"],
    },
    {
        "name": "Cynthia Garchomp",
        "any": ["Cynthia's Garchomp ex"],
    },
    {"name": "N's Zoroark", "any": ["N's Zoroark ex"]},
    {"name": "Mega Abomasnow", "any": ["Mega Abomasnow ex"]},
    {"name": "Mega Froslass", "any": ["Mega Froslass ex"]},
    {"name": "Mega Lucario", "any": ["Mega Lucario ex"]},
    {"name": "Archaludon", "any": ["Archaludon ex"]},
    {"name": "Crustle Wall", "any": ["Crustle"]},
    {"name": "Dragapult", "any": ["Dragapult ex"]},
    {"name": "Mega Starmie", "any": ["Mega Starmie ex"]},
    {"name": "Starmie", "any": ["Starmie ex", "Starmie"]},
    {"name": "Mega Gardevoir", "any": ["Mega Gardevoir ex"]},
    {"name": "Alakazam", "any": ["Alakazam ex", "Alakazam"]},
    {
        "name": "Iono Bellibolt",
        "any": ["Iono's Bellibolt ex"],
    },
    {"name": "Festival Lead", "any": ["Dipplin"]},
    {
        "name": "Hop Trevenant",
        "any": ["Hop's Trevenant"],
    },
    {"name": "Hop Snorlax", "any": ["Hop's Snorlax"]},
    {"name": "Mega Kangaskhan", "any": ["Mega Kangaskhan ex"]},
    {"name": "Chandelure", "any": ["Chandelure ex", "Chandelure"]},
    {"name": "Mega Greninja", "any": ["Mega Greninja ex"]},
    {"name": "Mega Clefable", "any": ["Mega Clefable ex"]},
    {
        "name": "Team Rocket Mewtwo",
        "any": ["Team Rocket's Mewtwo ex"],
    },
    {"name": "Comfey", "any": ["Comfey"]},
)

FALLBACK_ARCHETYPE_NAMES = {
    "Budew": "Budew",
    "Ceruledge ex": "Ceruledge",
    "Cornerstone Mask Ogerpon ex": "Cornerstone Mask Ogerpon",
    "Cubchoo": "Cubchoo",
    "Decidueye ex": "Decidueye",
    "Eevee": "Eevee",
    "Empoleon ex": "Empoleon",
    "Ethan's Cyndaquil": "Ethan's Cyndaquil",
    "Fezandipiti ex": "Fezandipiti",
    "Flareon ex": "Flareon",
    "Flygon ex": "Flygon",
    "Hydreigon ex": "Hydreigon",
    "Latias ex": "Latias",
    "Lillie's Clefairy ex": "Lillie's Clefairy",
    "Mega Lopunny ex": "Mega Lopunny",
    "Mega Sharpedo ex": "Mega Sharpedo",
    "Mega Zygarde ex": "Mega Zygarde",
    "Milotic ex": "Milotic",
    "Okidogi": "Okidogi",
    "Paldean Tauros": "Paldean Tauros",
    "Pikachu ex": "Pikachu",
    "Raging Bolt ex": "Raging Bolt",
    "Solrock": "Solrock",
    "Spheal": "Spheal",
    "Teal Mask Ogerpon ex": "Teal Mask Ogerpon",
    "Team Rocket's Arbok": "Team Rocket's Arbok",
    "Team Rocket's Chingling": "Team Rocket's Chingling",
    "Team Rocket's Honchkrow": "Team Rocket's Honchkrow",
    "Team Rocket's Murkrow": "Team Rocket's Murkrow",
}

REQUIRED_FACT_COLUMNS = {
    "date",
    "episode_id",
    "player",
    "deck",
    "reward",
    "result",
}


def normalize_card_name(name: Any) -> str:
    text = unicodedata.normalize("NFKC", str(name or ""))
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _validate_deck(deck: Iterable[int]) -> list[int]:
    values = list(deck)
    if len(values) != 60:
        raise ValueError(f"deck must contain exactly 60 cards, found {len(values)}")
    if any(type(card_id) is not int or card_id < 0 for card_id in values):
        raise ValueError("deck must contain non-negative integer Card IDs")
    return values


@dataclass(frozen=True, slots=True)
class CardCatalog:
    names: dict[int, str]
    kinds: dict[int, str]

    @classmethod
    def from_csv(cls, path: str | Path) -> "CardCatalog":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Card table not found: {path}")
        raw = pd.read_csv(path, encoding="utf-8-sig")
        if raw.shape[1] < 5:
            raise ValueError("Card table must contain at least five columns")
        card_ids = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
        valid = card_ids.notna()
        names = {
            int(card_id): str(name)
            for card_id, name in zip(card_ids[valid], raw.loc[valid].iloc[:, 1])
        }
        kinds = {
            int(card_id): str(kind)
            for card_id, kind in zip(card_ids[valid], raw.loc[valid].iloc[:, 4])
        }
        if not names:
            raise ValueError(f"Card table contains no valid Card IDs: {path}")
        return cls(names=names, kinds=kinds)

    def name(self, card_id: int) -> str:
        try:
            return self.names[int(card_id)]
        except KeyError as exc:
            raise ValueError(f"Unknown Card ID: {card_id}") from exc

    def is_pokemon(self, card_id: int) -> bool:
        self.name(card_id)
        return "pok" in normalize_card_name(self.kinds.get(int(card_id), ""))


@dataclass(frozen=True, slots=True)
class DeckClassification:
    archetype: str
    method: str
    evidence: str
    representative_card_id: int | None
    representative_card_name: str


@dataclass(frozen=True, slots=True)
class ExactDeckIdentity:
    deck_id: str
    full_hash: str
    card_ids: tuple[int, ...]


@dataclass(frozen=True)
class IsolationRoll:
    selected: tuple[str, ...]
    replay_count: int
    attempts: int


class IsolationSamplingError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _representative_card_id(
    deck: list[int],
    normalized_name: str,
    catalog: CardCatalog,
) -> int | None:
    candidates = Counter(
        card_id
        for card_id in deck
        if normalize_card_name(catalog.name(card_id)) == normalized_name
    )
    if not candidates:
        return None
    return sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[0][0]


def classify_deck(
    deck: Iterable[int],
    catalog: CardCatalog,
) -> DeckClassification:
    cards = _validate_deck(deck)
    normalized_to_display: dict[str, str] = {}
    name_counts: Counter[str] = Counter()
    for card_id in cards:
        display_name = catalog.name(card_id)
        normalized = normalize_card_name(display_name)
        normalized_to_display.setdefault(normalized, display_name)
        name_counts[normalized] += 1
    present = set(name_counts)

    for rule in ARCHETYPE_RULES:
        required_all = {
            normalize_card_name(name) for name in rule.get("all", [])
        }
        required_any = {
            normalize_card_name(name) for name in rule.get("any", [])
        }
        if required_all and not required_all.issubset(present):
            continue
        if required_any and not (required_any & present):
            continue
        ordered_markers = [
            normalize_card_name(name)
            for name in rule.get("all", []) + rule.get("any", [])
        ]
        matched = list(
            dict.fromkeys(name for name in ordered_markers if name in present)
        )
        representative_name = matched[0] if matched else ""
        representative_id = _representative_card_id(
            cards, representative_name, catalog
        )
        return DeckClassification(
            archetype=str(rule["name"]),
            method="rule",
            evidence=", ".join(
                normalized_to_display[name] for name in matched
            ),
            representative_card_id=representative_id,
            representative_card_name=(
                catalog.name(representative_id)
                if representative_id is not None
                else ""
            ),
        )

    pokemon_counts: Counter[str] = Counter()
    for card_id, count in Counter(cards).items():
        if catalog.is_pokemon(card_id):
            pokemon_counts[normalize_card_name(catalog.name(card_id))] += count
    if not pokemon_counts:
        return DeckClassification("Unknown", "no_pokemon_found", "", None, "")

    ex_candidates = [
        (count, name)
        for name, count in pokemon_counts.items()
        if name.endswith(" ex")
    ]
    candidates = ex_candidates or [
        (count, name) for name, count in pokemon_counts.items()
    ]
    _, chosen = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    chosen_display = normalized_to_display[chosen]
    normalized_aliases = {
        normalize_card_name(name): archetype
        for name, archetype in FALLBACK_ARCHETYPE_NAMES.items()
    }
    named_archetype = normalized_aliases.get(chosen)
    method = (
        "named_fallback_main_pokemon"
        if named_archetype is not None
        else "fallback_main_pokemon"
    )
    return DeckClassification(
        archetype=named_archetype or f"Other / {chosen_display}",
        method=method,
        evidence=chosen_display,
        representative_card_id=_representative_card_id(
            cards, chosen, catalog
        ),
        representative_card_name=chosen_display,
    )


def exact_deck_identity(deck: Iterable[int]) -> ExactDeckIdentity:
    cards = tuple(sorted(_validate_deck(deck)))
    canonical = ",".join(map(str, cards))
    full_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()
    return ExactDeckIdentity(
        deck_id=f"D-{full_hash[:12]}",
        full_hash=full_hash,
        card_ids=cards,
    )


def _parse_deck(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("deck contains invalid JSON") from exc
    if not isinstance(value, list):
        raise ValueError("deck must be a JSON list")
    return _validate_deck(value)


def validate_deck_rows(rows: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_FACT_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError(f"Deck rows are missing columns: {sorted(missing)}")
    facts = rows.copy().reset_index(drop=True)
    if facts.empty:
        raise ValueError("Deck rows are empty")
    facts["date"] = facts["date"].astype(str)
    facts["episode_id"] = facts["episode_id"].astype(str)
    if facts["date"].str.strip().isin({"", "nan", "none"}).any():
        raise ValueError("date must not be empty")
    if facts["episode_id"].str.strip().isin({"", "nan", "none"}).any():
        raise ValueError("episode_id must not be empty")
    players = pd.to_numeric(facts["player"], errors="raise")
    if not players.isin([0, 1]).all():
        raise ValueError("player must be 0 or 1")
    facts["player"] = players.astype(int)
    facts["reward"] = pd.to_numeric(facts["reward"], errors="raise").astype(float)
    facts["result"] = facts["result"].astype(str).str.strip().str.lower()
    invalid_results = sorted(set(facts["result"]) - {"win", "loss", "draw"})
    if invalid_results:
        raise ValueError(f"Invalid result labels: {invalid_results}")
    facts["deck"] = facts["deck"].map(_parse_deck)
    key_columns = ["date", "episode_id", "player"]
    if facts.duplicated(key_columns).any():
        raise ValueError("Duplicate (date, episode_id, player) rows found")
    episode_players = facts.groupby(["date", "episode_id"])["player"].agg(
        lambda values: tuple(sorted(values))
    )
    invalid_episodes = episode_players[
        episode_players.map(lambda players: players != (0, 1))
    ]
    if not invalid_episodes.empty:
        raise ValueError(
            f"{len(invalid_episodes)} episodes do not contain players 0 and 1"
        )
    return facts


def load_deck_rows(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for path_value in paths:
        path = Path(path_value)
        frame = pd.read_csv(
            path,
            dtype={"date": str, "episode_id": str},
        )
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError("No deck CSV files were provided")
    return validate_deck_rows(pd.concat(frames, ignore_index=True))


def annotate_deck_facts(
    rows: pd.DataFrame,
    catalog: CardCatalog,
) -> pd.DataFrame:
    """Add stable identities and archetypes to the two rows of each replay."""
    facts = validate_deck_rows(rows)
    identities = facts["deck"].map(exact_deck_identity)
    classifications = facts["deck"].map(
        lambda deck: classify_deck(deck, catalog)
    )
    facts["replay_key"] = (
        facts["date"].astype(str)
        + "\x1f"
        + facts["episode_id"].astype(str)
    )
    facts["deck_id"] = identities.map(lambda item: item.deck_id)
    facts["deck_card_ids"] = identities.map(lambda item: item.card_ids)
    facts["deck_archetype"] = classifications.map(
        lambda item: item.archetype
    )
    facts["classification_method"] = classifications.map(
        lambda item: item.method
    )
    facts["representative_card_id"] = classifications.map(
        lambda item: item.representative_card_id
    )
    return facts


def _require_annotated_facts(facts: pd.DataFrame) -> pd.DataFrame:
    required = {
        "replay_key",
        "deck_id",
        "deck_card_ids",
        "deck_archetype",
        "representative_card_id",
        "result",
    }
    missing = required - set(facts.columns)
    if missing:
        raise ValueError(
            f"Annotated deck facts are missing columns: {sorted(missing)}"
        )
    if facts.empty:
        raise ValueError("Annotated deck facts are empty")
    return facts


def _deck_indexes(
    facts: pd.DataFrame,
) -> tuple[
    dict[str, tuple[int, ...]],
    dict[str, str],
    dict[str, set[str]],
]:
    cards_by_deck: dict[str, tuple[int, ...]] = {}
    archetype_by_deck: dict[str, str] = {}
    replays_by_deck: dict[str, set[str]] = {}
    for row in facts.itertuples(index=False):
        deck_id = str(row.deck_id)
        cards_by_deck.setdefault(deck_id, tuple(row.deck_card_ids))
        archetype_by_deck.setdefault(deck_id, str(row.deck_archetype))
        replays_by_deck.setdefault(deck_id, set()).add(str(row.replay_key))
    return cards_by_deck, archetype_by_deck, replays_by_deck


def _training_deck_ids(
    replays_by_deck: dict[str, set[str]],
    validation_replays: set[str],
) -> set[str]:
    return {
        deck_id
        for deck_id, replay_keys in replays_by_deck.items()
        if replay_keys - validation_replays
    }


def _card_ids_in_decks(
    deck_ids: Iterable[str],
    cards_by_deck: dict[str, tuple[int, ...]],
) -> set[int]:
    return {
        card_id
        for deck_id in deck_ids
        for card_id in cards_by_deck[deck_id]
    }


def _assign_similarity_bands(
    frame: pd.DataFrame,
    distance_column: str,
) -> pd.DataFrame:
    """Assign tie-preserving empirical thirds to eligible rows."""
    result = frame.copy()
    result["similarity_band"] = "not_eligible"
    eligible_mask = result["eligible"].fillna(False) & result[
        distance_column
    ].notna()
    distances = result.loc[eligible_mask, distance_column].astype(float)
    if distances.empty:
        result.attrs["similarity_bands"] = {
            "q33": None,
            "q67": None,
            "counts": {},
        }
        return result

    q33 = float(distances.quantile(1 / 3, interpolation="nearest"))
    q67 = float(distances.quantile(2 / 3, interpolation="nearest"))

    def similarity_band(distance: float) -> str:
        if distance <= q33:
            return "high"
        if q33 < q67 and distance <= q67:
            return "moderate"
        return "lower"

    result.loc[eligible_mask, "similarity_band"] = distances.map(
        similarity_band
    )
    counts = (
        result.loc[eligible_mask, "similarity_band"]
        .value_counts()
        .reindex(["high", "moderate", "lower"], fill_value=0)
    )
    result.attrs["similarity_bands"] = {
        "q33": q33,
        "q67": q67,
        "counts": {
            str(band): int(count)
            for band, count in counts.items()
            if count > 0
        },
    }
    return result


def build_deck_isolation_candidates(
    annotated_facts: pd.DataFrame,
    min_validation_replays: int = 100,
    excluded_archetypes: Sequence[str] = (),
) -> pd.DataFrame:
    """Evaluate every exact deck as a hypothetical replay-level holdout."""
    if min_validation_replays < 1:
        raise ValueError("min_validation_replays must be >= 1")

    facts = _require_annotated_facts(annotated_facts)
    cards_by_deck, archetype_by_deck, replays_by_deck = _deck_indexes(facts)
    excluded = set(map(str, excluded_archetypes))
    all_replays = set(facts["replay_key"].astype(str))
    uses = facts.groupby("deck_id").size().to_dict()
    wins = (
        facts["result"].eq("win").groupby(facts["deck_id"]).sum().to_dict()
    )
    replay_uses_by_deck: dict[str, Counter[str]] = {}
    for row in facts.itertuples(index=False):
        replay_uses_by_deck.setdefault(
            str(row.deck_id), Counter()
        )[str(row.replay_key)] += 1
    rows: list[dict[str, Any]] = []

    for deck_id in sorted(cards_by_deck):
        validation_replays = replays_by_deck[deck_id]
        train_deck_ids = _training_deck_ids(
            replays_by_deck, validation_replays
        )
        candidate_cards = cards_by_deck[deck_id]
        archetype = archetype_by_deck[deck_id]
        same_archetype_ids = sorted(
            other_id
            for other_id in train_deck_ids
            if other_id != deck_id
            and archetype_by_deck[other_id] == archetype
        )
        training_card_ids = _card_ids_in_decks(
            train_deck_ids, cards_by_deck
        )
        unseen_card_ids = sorted(set(candidate_cards) - training_card_ids)
        # Validation contains every replay in which this exact deck appears.
        exact_deck_train_uses = 0

        nearest_id: str | None = None
        nearest_slots: int | None = None
        nearest_jaccard: float | None = None
        reference_id: str | None = None
        reference_train_uses: int | None = None
        reference_slots: int | None = None
        reference_jaccard: float | None = None
        if same_archetype_ids:
            training_uses = {
                other_id: sum(
                    count
                    for replay_key, count in replay_uses_by_deck[
                        other_id
                    ].items()
                    if replay_key not in validation_replays
                )
                for other_id in same_archetype_ids
            }
            reference_id = min(
                same_archetype_ids,
                key=lambda other_id: (
                    -training_uses[other_id],
                    other_id,
                ),
            )
            reference_train_uses = training_uses[reference_id]
            reference_slots = changed_slots(
                candidate_cards, cards_by_deck[reference_id]
            )
            reference_jaccard = weighted_jaccard(
                candidate_cards, cards_by_deck[reference_id]
            )
            distances = [
                (
                    other_id,
                    changed_slots(
                        candidate_cards, cards_by_deck[other_id]
                    ),
                    weighted_jaccard(
                        candidate_cards, cards_by_deck[other_id]
                    ),
                )
                for other_id in same_archetype_ids
            ]
            nearest_id, nearest_slots, nearest_jaccard = min(
                distances,
                key=lambda item: (item[1], -item[2], item[0]),
            )

        replay_count = len(validation_replays)
        meets_min_replays = replay_count >= min_validation_replays
        all_cards_seen = not unseen_card_ids
        eligible = (
            meets_min_replays
            and bool(same_archetype_ids)
            and all_cards_seen
            and exact_deck_train_uses == 0
            and len(all_replays - validation_replays) > 0
            and archetype not in excluded
        )
        deck_uses = int(uses[deck_id])
        deck_wins = int(wins.get(deck_id, 0))
        rows.append(
            {
                "deck_id": deck_id,
                "deck_archetype": archetype,
                "replays": replay_count,
                "uses": deck_uses,
                "wins": deck_wins,
                "win_rate": deck_wins / deck_uses,
                "remaining_training_replays": len(
                    all_replays - validation_replays
                ),
                "remaining_archetype_decks": len(same_archetype_ids),
                "all_cards_seen_in_train": all_cards_seen,
                "unseen_card_ids_json": _compact_json(unseen_card_ids),
                "exact_deck_train_uses": exact_deck_train_uses,
                "nearest_train_deck_id": nearest_id,
                "nearest_changed_slots": nearest_slots,
                "nearest_weighted_jaccard": nearest_jaccard,
                "reference_train_deck_id": reference_id,
                "reference_train_uses": reference_train_uses,
                "reference_changed_slots": reference_slots,
                "reference_weighted_jaccard": reference_jaccard,
                "meets_min_replays": meets_min_replays,
                "excluded_archetype": archetype in excluded,
                "eligible": eligible,
            }
        )

    result = _assign_similarity_bands(
        pd.DataFrame(rows),
        "reference_changed_slots",
    )
    band_info = result.attrs["similarity_bands"]
    result = result.sort_values(
        ["eligible", "replays", "deck_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result.attrs["similarity_bands"] = band_info
    return result


def build_top_deck_archetype_candidates(
    annotated_facts: pd.DataFrame,
    target_archetype: str,
    top_deck_card_ids: Iterable[int],
    min_validation_replays: int = 100,
) -> pd.DataFrame:
    """Build holdout candidates within one archetype around an exact deck."""
    facts = _require_annotated_facts(annotated_facts)
    target = str(target_archetype).strip()
    if not target:
        raise ValueError("target_archetype must not be empty")
    if target not in set(facts["deck_archetype"].astype(str)):
        raise ValueError(f"Unknown target archetype: {target}")

    top_identity = exact_deck_identity(top_deck_card_ids)
    candidates = build_deck_isolation_candidates(
        facts,
        min_validation_replays=min_validation_replays,
    )
    candidates = candidates[
        candidates["deck_archetype"].astype(str).eq(target)
    ].copy()
    cards_by_deck, _, _ = _deck_indexes(facts)
    candidates["is_top_deck"] = candidates["deck_id"].astype(str).eq(
        top_identity.deck_id
    )
    candidates["eligible"] = (
        candidates["eligible"] & ~candidates["is_top_deck"]
    )
    candidates["top_deck_changed_slots"] = candidates["deck_id"].map(
        lambda deck_id: changed_slots(
            cards_by_deck[str(deck_id)],
            top_identity.card_ids,
        )
    )
    candidates["top_deck_weighted_jaccard"] = candidates["deck_id"].map(
        lambda deck_id: weighted_jaccard(
            cards_by_deck[str(deck_id)],
            top_identity.card_ids,
        )
    )
    candidates = _assign_similarity_bands(
        candidates,
        "top_deck_changed_slots",
    )
    band_info = candidates.attrs["similarity_bands"]
    candidates = candidates.sort_values(
        ["eligible", "top_deck_changed_slots", "replays", "deck_id"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    candidates.attrs["similarity_bands"] = {
        **band_info,
        "top_deck_id": top_identity.deck_id,
        "top_deck_present": bool(candidates["is_top_deck"].any()),
    }
    return candidates


def _rule_marker_names(archetype: str) -> set[str]:
    for rule in ARCHETYPE_RULES:
        if rule["name"] == archetype:
            return {
                normalize_card_name(name)
                for name in rule.get("all", []) + rule.get("any", [])
            }
    return set()


def archetype_core_card_ids(
    annotated_facts: pd.DataFrame,
    catalog: CardCatalog,
    archetype: str,
) -> tuple[int, ...]:
    """Resolve core Card IDs actually observed for one archetype."""
    facts = _require_annotated_facts(annotated_facts)
    archetype_rows = facts[facts["deck_archetype"] == archetype]
    if archetype_rows.empty:
        raise ValueError(f"Unknown archetype: {archetype}")

    marker_names = _rule_marker_names(archetype)
    core_ids: set[int] = set()
    if marker_names:
        for cards in archetype_rows["deck_card_ids"]:
            core_ids.update(
                card_id
                for card_id in set(cards)
                if normalize_card_name(catalog.name(card_id))
                in marker_names
            )
    else:
        core_ids.update(
            int(card_id)
            for card_id in archetype_rows["representative_card_id"].dropna()
        )
    return tuple(sorted(core_ids))


def _validate_roll_constraints(
    min_count: int,
    total_replays_min: int,
    total_replays_max: int,
    max_attempts: int,
) -> None:
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    if total_replays_min < 0 or total_replays_min > total_replays_max:
        raise ValueError(
            "total replay range must satisfy 0 <= min <= max"
        )
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")


def roll_deck_isolation_selection(
    annotated_facts: pd.DataFrame,
    candidates: pd.DataFrame,
    min_count: int,
    total_replays_min: int,
    total_replays_max: int,
    roll_id: int,
    max_attempts: int = 2000,
    excluded_archetypes: Sequence[str] = (),
) -> IsolationRoll:
    """Roll exact decks with unique archetypes and equal band weights."""
    _validate_roll_constraints(
        min_count,
        total_replays_min,
        total_replays_max,
        max_attempts,
    )
    required = {
        "deck_id",
        "deck_archetype",
        "similarity_band",
        "eligible",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            f"Deck candidates are missing columns: {sorted(missing)}"
        )
    facts = _require_annotated_facts(annotated_facts)
    _, _, replays_by_deck = _deck_indexes(facts)
    eligible = candidates[candidates["eligible"]].copy()
    eligible["deck_id"] = eligible["deck_id"].astype(str)
    eligible["deck_archetype"] = eligible["deck_archetype"].astype(str)
    eligible["similarity_band"] = eligible["similarity_band"].astype(str)
    excluded = set(map(str, excluded_archetypes))
    eligible = eligible[
        ~eligible["deck_archetype"].isin(excluded)
    ].copy()
    candidate_replay_counts = [
        len(replays_by_deck[deck_id])
        for deck_id in eligible["deck_id"]
    ]
    diagnostics = {
        "eligible_decks": len(eligible),
        "available_archetypes": int(
            eligible["deck_archetype"].nunique()
        ),
        "band_counts": {
            str(key): int(value)
            for key, value in eligible["similarity_band"]
            .value_counts()
            .to_dict()
            .items()
        },
        "min_count": min_count,
        "total_replays_min": total_replays_min,
        "total_replays_max": total_replays_max,
        "excluded_archetypes": sorted(excluded),
        "candidate_replays_min": min(
            candidate_replay_counts, default=0
        ),
        "candidate_replays_max": max(
            candidate_replay_counts, default=0
        ),
    }
    if diagnostics["available_archetypes"] < min_count:
        raise IsolationSamplingError(
            "Not enough unique eligible archetypes for Deck Isolation. "
            "Lower DECK_MIN_COUNT or DECK_MIN_REPLAYS.",
            diagnostics,
        )

    rng = random.Random(int(roll_id))
    for attempt in range(1, max_attempts + 1):
        selected_ids: list[str] = []
        selected_archetypes: set[str] = set()
        replay_union: set[str] = set()
        while True:
            if (
                len(selected_ids) >= min_count
                and total_replays_min
                <= len(replay_union)
                <= total_replays_max
            ):
                return IsolationRoll(
                    selected=tuple(selected_ids),
                    replay_count=len(replay_union),
                    attempts=attempt,
                )

            available = eligible[
                ~eligible["deck_archetype"].isin(selected_archetypes)
                & ~eligible["deck_id"].isin(selected_ids)
            ]
            if available.empty:
                break
            feasible_mask = available["deck_id"].map(
                lambda deck_id: len(
                    replay_union | replays_by_deck[str(deck_id)]
                )
                <= total_replays_max
            )
            feasible = available[feasible_mask]
            if feasible.empty:
                break

            bands = sorted(feasible["similarity_band"].unique())
            band = rng.choice(bands)
            band_rows = feasible[
                feasible["similarity_band"] == band
            ]
            archetype = rng.choice(
                sorted(band_rows["deck_archetype"].unique())
            )
            deck_id = rng.choice(
                sorted(
                    band_rows.loc[
                        band_rows["deck_archetype"] == archetype,
                        "deck_id",
                    ]
                )
            )
            selected_ids.append(str(deck_id))
            selected_archetypes.add(str(archetype))
            replay_union.update(replays_by_deck[str(deck_id)])

    raise IsolationSamplingError(
        f"No feasible Deck Isolation roll after {max_attempts} attempts. "
        "Lower the minimums or widen the total replay range.",
        diagnostics,
    )


def roll_top_deck_archetype_selection(
    annotated_facts: pd.DataFrame,
    candidates: pd.DataFrame,
    min_count: int,
    total_replays_min: int,
    total_replays_max: int,
    roll_id: int,
    max_attempts: int = 2000,
) -> IsolationRoll:
    """Roll target-archetype variants with equal available-band weights."""
    _validate_roll_constraints(
        min_count,
        total_replays_min,
        total_replays_max,
        max_attempts,
    )
    required = {
        "deck_id",
        "deck_archetype",
        "similarity_band",
        "eligible",
        "is_top_deck",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            "Top-deck candidates are missing columns: "
            f"{sorted(missing)}"
        )
    facts = _require_annotated_facts(annotated_facts)
    _, _, replays_by_deck = _deck_indexes(facts)
    eligible = candidates[
        candidates["eligible"] & ~candidates["is_top_deck"]
    ].copy()
    eligible["deck_id"] = eligible["deck_id"].astype(str)
    eligible["similarity_band"] = eligible["similarity_band"].astype(str)
    candidate_replay_counts = [
        len(replays_by_deck[deck_id])
        for deck_id in eligible["deck_id"]
    ]
    diagnostics = {
        "eligible_decks": len(eligible),
        "band_counts": {
            str(key): int(value)
            for key, value in eligible["similarity_band"]
            .value_counts()
            .to_dict()
            .items()
        },
        "min_count": min_count,
        "total_replays_min": total_replays_min,
        "total_replays_max": total_replays_max,
        "candidate_replays_min": min(
            candidate_replay_counts, default=0
        ),
        "candidate_replays_max": max(
            candidate_replay_counts, default=0
        ),
    }
    if len(eligible) < min_count:
        raise IsolationSamplingError(
            "Not enough eligible variants in the top-deck archetype. "
            "Lower TOP_DECK_MIN_COUNT or TOP_DECK_MIN_REPLAYS.",
            diagnostics,
        )

    rng = random.Random(int(roll_id))
    for attempt in range(1, max_attempts + 1):
        selected: list[str] = []
        replay_union: set[str] = set()
        while True:
            if (
                len(selected) >= min_count
                and total_replays_min
                <= len(replay_union)
                <= total_replays_max
            ):
                return IsolationRoll(
                    selected=tuple(selected),
                    replay_count=len(replay_union),
                    attempts=attempt,
                )
            available = eligible[~eligible["deck_id"].isin(selected)]
            if available.empty:
                break
            feasible_mask = available["deck_id"].map(
                lambda deck_id: len(
                    replay_union | replays_by_deck[str(deck_id)]
                )
                <= total_replays_max
            )
            feasible = available[feasible_mask]
            if feasible.empty:
                break
            band = rng.choice(
                sorted(feasible["similarity_band"].unique())
            )
            band_rows = feasible[
                feasible["similarity_band"] == band
            ]
            deck_id = rng.choice(sorted(band_rows["deck_id"]))
            selected.append(str(deck_id))
            replay_union.update(replays_by_deck[str(deck_id)])

    raise IsolationSamplingError(
        f"No feasible top-deck archetype roll after {max_attempts} "
        "attempts. Lower the minimums or widen the total replay range.",
        diagnostics,
    )


def build_archetype_isolation_candidates(
    annotated_facts: pd.DataFrame,
    catalog: CardCatalog,
    min_validation_replays: int = 100,
) -> pd.DataFrame:
    """Aggregate primary archetypes with curated sampling eligibility."""
    if min_validation_replays < 1:
        raise ValueError("min_validation_replays must be >= 1")
    facts = _require_annotated_facts(annotated_facts)
    cards_by_deck, archetype_by_deck, replays_by_deck = _deck_indexes(facts)
    rows: list[dict[str, Any]] = []

    for archetype in sorted(set(archetype_by_deck.values())):
        archetype_rows = facts[facts["deck_archetype"] == archetype]
        archetype_decks = set(archetype_rows["deck_id"].astype(str))
        replays = set(archetype_rows["replay_key"].astype(str))
        core_ids = set(
            archetype_core_card_ids(facts, catalog, archetype)
        )
        overlap_decks = {
            deck_id
            for deck_id, cards in cards_by_deck.items()
            if archetype_by_deck[deck_id] != archetype
            and bool(core_ids.intersection(cards))
        }
        overlap_replays = {
            replay_key
            for deck_id in overlap_decks
            for replay_key in replays_by_deck[deck_id]
        }
        uses = len(archetype_rows)
        wins = int(archetype_rows["result"].eq("win").sum())
        rule_defined = bool(_rule_marker_names(archetype))
        meets_min = len(replays) >= min_validation_replays
        methods = sorted(
            set(archetype_rows["classification_method"].astype(str))
        )
        sampling_method_allowed = bool(methods) and set(methods).issubset(
            {"rule", "named_fallback_main_pokemon"}
        )
        rows.append(
            {
                "deck_archetype": archetype,
                "classification_method": ", ".join(methods),
                "core_card_ids_json": _compact_json(sorted(core_ids)),
                "core_card_names": ", ".join(
                    sorted({catalog.name(card_id) for card_id in core_ids})
                ),
                "replays": len(replays),
                "uses": uses,
                "wins": wins,
                "win_rate": wins / uses,
                "unique_exact_decks": len(archetype_decks),
                "core_card_other_archetype_decks": len(overlap_decks),
                "core_card_other_archetype_replays": len(
                    overlap_replays
                ),
                "rule_defined": rule_defined,
                "sampling_method_allowed": sampling_method_allowed,
                "meets_min_replays": meets_min,
                "eligible": sampling_method_allowed and meets_min,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["eligible", "replays", "deck_archetype"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def roll_archetype_isolation_selection(
    annotated_facts: pd.DataFrame,
    candidates: pd.DataFrame,
    min_count: int,
    total_replays_min: int,
    total_replays_max: int,
    roll_id: int,
    max_attempts: int = 2000,
    excluded_archetypes: Sequence[str] = (),
) -> IsolationRoll:
    """Roll rule-defined archetypes uniformly without replacement."""
    _validate_roll_constraints(
        min_count,
        total_replays_min,
        total_replays_max,
        max_attempts,
    )
    required = {"deck_archetype", "eligible"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            f"Archetype candidates are missing columns: {sorted(missing)}"
        )
    facts = _require_annotated_facts(annotated_facts)
    eligible = candidates[candidates["eligible"]].copy()
    excluded = set(map(str, excluded_archetypes))
    eligible = eligible[
        ~eligible["deck_archetype"].astype(str).isin(excluded)
    ].copy()
    archetypes = sorted(
        eligible["deck_archetype"].astype(str).unique()
    )
    replays_by_archetype = {
        archetype: set(
            facts.loc[
                facts["deck_archetype"] == archetype,
                "replay_key",
            ].astype(str)
        )
        for archetype in archetypes
    }
    diagnostics = {
        "eligible_archetypes": len(archetypes),
        "min_count": min_count,
        "total_replays_min": total_replays_min,
        "total_replays_max": total_replays_max,
        "excluded_archetypes": sorted(excluded),
        "candidate_replays_min": int(
            eligible["replays"].min()
        ) if not eligible.empty and "replays" in eligible else 0,
        "candidate_replays_max": int(
            eligible["replays"].max()
        ) if not eligible.empty and "replays" in eligible else 0,
    }
    if len(archetypes) < min_count:
        raise IsolationSamplingError(
            "Not enough eligible rule-defined archetypes. Lower "
            "ARCHETYPE_MIN_COUNT or ARCHETYPE_MIN_REPLAYS.",
            diagnostics,
        )

    rng = random.Random(int(roll_id))
    for attempt in range(1, max_attempts + 1):
        selected: list[str] = []
        replay_union: set[str] = set()
        while True:
            if (
                len(selected) >= min_count
                and total_replays_min
                <= len(replay_union)
                <= total_replays_max
            ):
                return IsolationRoll(
                    selected=tuple(selected),
                    replay_count=len(replay_union),
                    attempts=attempt,
                )
            feasible = [
                archetype
                for archetype in archetypes
                if archetype not in selected
                and len(
                    replay_union | replays_by_archetype[archetype]
                )
                <= total_replays_max
            ]
            if not feasible:
                break
            archetype = rng.choice(sorted(feasible))
            selected.append(archetype)
            replay_union.update(replays_by_archetype[archetype])

    raise IsolationSamplingError(
        f"No feasible Archetype Isolation roll after {max_attempts} attempts. "
        "Lower the minimums or widen the total replay range.",
        diagnostics,
    )


def build_archetype_selection_details(
    annotated_facts: pd.DataFrame,
    selected_archetypes: Sequence[str],
) -> pd.DataFrame:
    """Expand selected archetypes into exact-deck rows for later splitting."""
    facts = _require_annotated_facts(annotated_facts)
    selected = tuple(dict.fromkeys(map(str, selected_archetypes)))
    unknown = sorted(set(selected) - set(facts["deck_archetype"]))
    if unknown:
        raise ValueError(f"Unknown selected archetypes: {unknown}")
    selected_rows = facts[facts["deck_archetype"].isin(selected)]
    rows: list[dict[str, Any]] = []
    for deck_id, group in selected_rows.groupby("deck_id", sort=False):
        uses = len(group)
        wins = int(group["result"].eq("win").sum())
        rows.append(
            {
                "deck_archetype": str(group.iloc[0]["deck_archetype"]),
                "deck_id": str(deck_id),
                "replays": int(group["replay_key"].nunique()),
                "uses": uses,
                "wins": wins,
                "win_rate": wins / uses,
                "card_ids": _compact_json(
                    list(group.iloc[0]["deck_card_ids"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["deck_archetype", "uses", "deck_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def audit_deck_archetype_selection(
    annotated_facts: pd.DataFrame,
    selected_deck_ids: Sequence[str],
    selected_archetypes: Sequence[str],
) -> dict[str, Any]:
    """Audit exact-deck and primary-archetype selections together."""
    facts = _require_annotated_facts(annotated_facts)
    cards_by_deck, archetype_by_deck, replays_by_deck = _deck_indexes(facts)
    selected_decks = tuple(dict.fromkeys(map(str, selected_deck_ids)))
    selected_arches = tuple(dict.fromkeys(map(str, selected_archetypes)))
    unknown_decks = sorted(set(selected_decks) - set(cards_by_deck))
    unknown_arches = sorted(
        set(selected_arches) - set(archetype_by_deck.values())
    )
    if unknown_decks:
        raise ValueError(f"Unknown selected deck IDs: {unknown_decks}")
    if unknown_arches:
        raise ValueError(f"Unknown selected archetypes: {unknown_arches}")

    deck_replays = {
        replay_key
        for deck_id in selected_decks
        for replay_key in replays_by_deck[deck_id]
    }
    archetype_replays = set(
        facts.loc[
            facts["deck_archetype"].isin(selected_arches),
            "replay_key",
        ].astype(str)
    )
    validation_replays = deck_replays | archetype_replays
    all_replays = set(facts["replay_key"].astype(str))
    train_replays = all_replays - validation_replays
    train_decks = _training_deck_ids(
        replays_by_deck, validation_replays
    )
    train_card_ids = _card_ids_in_decks(train_decks, cards_by_deck)

    exact_deck_train_uses = int(
        (
            facts["deck_id"].isin(selected_decks)
            & facts["replay_key"].astype(str).isin(train_replays)
        ).sum()
    )
    selected_deck_cards = {
        card_id
        for deck_id in selected_decks
        for card_id in cards_by_deck[deck_id]
    }
    missing_deck_card_ids = sorted(selected_deck_cards - train_card_ids)
    missing_deck_archetypes = sorted(
        {
            archetype_by_deck[deck_id]
            for deck_id in selected_decks
            if not any(
                archetype_by_deck[train_id]
                == archetype_by_deck[deck_id]
                for train_id in train_decks
            )
        }
    )
    archetype_train_uses = int(
        (
            facts["deck_archetype"].isin(selected_arches)
            & facts["replay_key"].astype(str).isin(train_replays)
        ).sum()
    )
    conflicting_archetypes = sorted(
        {
            archetype_by_deck[deck_id] for deck_id in selected_decks
        }.intersection(selected_arches)
    )
    valid = (
        exact_deck_train_uses == 0
        and not missing_deck_card_ids
        and not missing_deck_archetypes
        and archetype_train_uses == 0
        and not conflicting_archetypes
        and bool(train_replays)
    )
    return {
        "valid": valid,
        "total_replays": len(all_replays),
        "deck_isolation_replays": len(deck_replays),
        "archetype_isolation_replays": len(archetype_replays),
        "component_overlap_replays": len(
            deck_replays & archetype_replays
        ),
        "validation_replays": len(validation_replays),
        "remaining_training_replays": len(train_replays),
        "exact_deck_train_uses": exact_deck_train_uses,
        "missing_deck_card_ids": missing_deck_card_ids,
        "missing_deck_archetypes": missing_deck_archetypes,
        "archetype_train_uses": archetype_train_uses,
        "conflicting_archetypes": conflicting_archetypes,
    }


def _date_key(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d{1,2})\.(\d{1,2})\s*", str(value))
    if match is None:
        raise ValueError(f"Invalid month.day date: {value}")
    month, day = map(int, match.groups())
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        raise ValueError(f"Invalid month.day date: {value}")
    return month, day


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_deck_census(
    rows: pd.DataFrame,
    catalog: CardCatalog,
) -> pd.DataFrame:
    facts = validate_deck_rows(rows)
    identities = facts["deck"].map(exact_deck_identity)
    facts["deck_id"] = identities.map(lambda identity: identity.deck_id)
    facts["deck_hash_sha256"] = identities.map(
        lambda identity: identity.full_hash
    )
    facts["sorted_deck"] = identities.map(
        lambda identity: list(identity.card_ids)
    )

    canonical_by_id: dict[str, tuple[int, ...]] = {}
    for identity in identities:
        previous = canonical_by_id.setdefault(identity.deck_id, identity.card_ids)
        if previous != identity.card_ids:
            raise ValueError(f"Deck ID collision detected: {identity.deck_id}")

    opponent_ids = pd.Series(index=facts.index, dtype=object)
    for _, group in facts.groupby(["date", "episode_id"], sort=False):
        first_index, second_index = group.index.tolist()
        opponent_ids.at[first_index] = facts.at[second_index, "deck_id"]
        opponent_ids.at[second_index] = facts.at[first_index, "deck_id"]
    facts["opponent_deck_id"] = opponent_ids

    rows_out: list[dict[str, Any]] = []
    total_uses = len(facts)
    for deck_id, group in facts.groupby("deck_id", sort=False):
        first = group.iloc[0]
        deck = list(first["sorted_deck"])
        classification = classify_deck(deck, catalog)
        result_counts = Counter(group["result"])
        dates = sorted(set(group["date"]), key=_date_key)
        card_counts = Counter(deck)
        uses = len(group)
        rows_out.append(
            {
                "deck_archetype": classification.archetype,
                "classification_method": classification.method,
                "classification_evidence": classification.evidence,
                "representative_card_id": classification.representative_card_id,
                "representative_card_name":
                    classification.representative_card_name,
                "deck_id": deck_id,
                "deck_hash_sha256": first["deck_hash_sha256"],
                "uses": uses,
                "usage_percent": 100.0 * uses / total_uses,
                "wins": int(result_counts["win"]),
                "losses": int(result_counts["loss"]),
                "draws": int(result_counts["draw"]),
                "win_rate": result_counts["win"] / uses,
                "first_date": dates[0],
                "last_date": dates[-1],
                "unique_opponent_decks": int(
                    group["opponent_deck_id"].nunique()
                ),
                "deck_card_ids_json": _compact_json(deck),
                "deck_card_counts_json": _compact_json(
                    {
                        str(card_id): count
                        for card_id, count in sorted(card_counts.items())
                    }
                ),
            }
        )
    summary = pd.DataFrame(rows_out).sort_values(
        ["uses", "deck_id"],
        ascending=[False, True],
    )
    summary = summary.reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def changed_slots(deck1: Iterable[int], deck2: Iterable[int]) -> int:
    counts1 = Counter(_validate_deck(deck1))
    counts2 = Counter(_validate_deck(deck2))
    difference = sum(
        abs(counts1[card_id] - counts2[card_id])
        for card_id in counts1.keys() | counts2.keys()
    )
    if difference % 2:
        raise ValueError("equal-size decks must have an even L1 difference")
    return difference // 2


def weighted_jaccard(
    deck1: Iterable[int],
    deck2: Iterable[int],
) -> float:
    counts1 = Counter(_validate_deck(deck1))
    counts2 = Counter(_validate_deck(deck2))
    card_ids = counts1.keys() | counts2.keys()
    intersection = sum(
        min(counts1[card_id], counts2[card_id]) for card_id in card_ids
    )
    union = sum(
        max(counts1[card_id], counts2[card_id]) for card_id in card_ids
    )
    return intersection / union


def build_similarity_pairs(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "deck_id",
        "deck_archetype",
        "uses",
        "deck_card_ids_json",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Deck summary is missing columns: {sorted(missing)}")
    records = summary.to_dict("records")
    rows = []
    for first, second in combinations(records, 2):
        deck1 = _parse_deck(first["deck_card_ids_json"])
        deck2 = _parse_deck(second["deck_card_ids_json"])
        rows.append(
            {
                "deck_id_1": first["deck_id"],
                "deck_id_2": second["deck_id"],
                "archetype_1": first["deck_archetype"],
                "archetype_2": second["deck_archetype"],
                "same_archetype":
                    first["deck_archetype"] == second["deck_archetype"],
                "uses_1": int(first["uses"]),
                "uses_2": int(second["uses"]),
                "changed_slots": changed_slots(deck1, deck2),
                "weighted_jaccard": weighted_jaccard(deck1, deck2),
            }
        )
    pairs = pd.DataFrame(
        rows,
        columns=[
            "deck_id_1",
            "deck_id_2",
            "archetype_1",
            "archetype_2",
            "same_archetype",
            "uses_1",
            "uses_2",
            "changed_slots",
            "weighted_jaccard",
        ],
    )
    expected = len(records) * (len(records) - 1) // 2
    if len(pairs) != expected:
        raise RuntimeError(f"Pair count {len(pairs)} != expected {expected}")
    if not pairs.empty:
        if not pairs["changed_slots"].between(0, 60).all():
            raise RuntimeError("changed_slots is outside [0, 60]")
        if not pairs["weighted_jaccard"].between(0, 1).all():
            raise RuntimeError("weighted_jaccard is outside [0, 1]")
        pairs = pairs.sort_values(
            [
                "changed_slots",
                "weighted_jaccard",
                "deck_id_1",
                "deck_id_2",
            ],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)
    return pairs
