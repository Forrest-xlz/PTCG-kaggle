"""Sparse feature construction adapted from the competition notebook.

The public functions accept the typed objects returned by
``cg.api.to_observation_class``.  Keeping this module separate makes feature
parity testable and prevents extraction from depending on PyTorch.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from model.card_features import (
    CARD_ENERGY_TYPE_OFFSET,
    CARD_FEATURE_DIM,
    CARD_HP_INDEX,
    CARD_RESISTANCE_DIM,
    CARD_RESISTANCE_OFFSET,
    CARD_RETREAT_INDEX,
    CARD_SPECIAL_OFFSET,
    CARD_STAGE_OFFSET,
    CARD_TYPE_DIM,
    CARD_TYPE_OFFSET,
    CARD_WEAKNESS_DIM,
    CARD_WEAKNESS_OFFSET,
    ENERGY_TYPE_DIM,
    build_card_feature_table,
)


ENCODER_TOKENS = 26
POKEMON_ENCODER_TOKENS = 18
OWN_SUMMARY_DIM = 69
OPPONENT_SUMMARY_DIM = 71
GLOBAL_SUMMARY_DIM = 73
SELECT_TYPE_DIM = 11
SELECT_CONTEXT_DIM = 49
OPTION_CATEGORICAL_DIM = 11
OPTION_NUMERIC_DIM = 5
OPTION_TYPE_DIM = 17
OPTION_VALUE_MAX = 60
OPTION_VALUE_DIM = OPTION_VALUE_MAX + 3
OPTION_PLAYER_RELATION_DIM = 3
OPTION_AREA_DIM = 13
OPTION_SPECIAL_CONDITION_DIM = 6
POKEMON_DYNAMIC_WORD_DIM = 23
POKEMON_DYNAMIC_DIM = 2 * POKEMON_DYNAMIC_WORD_DIM
ATTACK_DYNAMIC_DIM = 6
HISTORY_STEPS = 3
HISTORY_STRUCTURAL_DIM = 8

OPTION_TYPE_INDEX = 0
OPTION_CONTEXT_INDEX = 1
OPTION_CANDIDATE_INDEX = 2
OPTION_TARGET_INDEX = 3
OPTION_ATTACK_INDEX = 4
OPTION_NUMBER_INDEX = 5
OPTION_COUNT_INDEX = 6
OPTION_PLAYER_RELATION_INDEX = 7
OPTION_AREA_INDEX = 8
OPTION_IN_PLAY_AREA_INDEX = 9
OPTION_SPECIAL_CONDITION_INDEX = 10


@dataclass
class SparseVector:
    index: list[int] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    offset: list[int] = field(default_factory=list)
    pos: int = 0

    def add(self, index: int, value: float) -> None:
        if float(value) != 0.0:
            self.index.append(self.pos + int(index)); self.value.append(float(value))
    def add_pos(self, count: int) -> None: self.pos += count
    def add_single(self, value: float) -> None:
        self.add(0, value); self.pos += 1
    def word_start(self) -> None: self.offset.append(len(self.index))


@dataclass(frozen=True)
class NumericFeatureCatalog:
    card_features: np.ndarray
    attack_damage: np.ndarray
    card_attacks: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.card_features.ndim != 2 or self.card_features.shape[1] != CARD_FEATURE_DIM:
            raise ValueError(
                f"card_features must have shape (card_count, {CARD_FEATURE_DIM})"
            )
        if len(self.card_attacks) != len(self.card_features):
            raise ValueError("card_attacks must align with card_features")


@dataclass(frozen=True)
class EncoderFeatures:
    sparse: SparseVector
    pokemon_appear: list[int]
    own_summary: list[float]
    opponent_summary: list[float]
    global_summary: list[float]


@dataclass(frozen=True)
class OptionFeatures:
    categorical: np.ndarray
    numeric: np.ndarray
    pokemon_dynamic: np.ndarray
    attack_dynamic: np.ndarray
    action_index: np.ndarray
    action_offset: np.ndarray


@dataclass(frozen=True)
class HistoryActionFeatures:
    """Selected-option features for one completed historical decision."""

    select_type: int
    select_context: int
    option_categorical: np.ndarray
    structural: np.ndarray
    pokemon_dynamic: np.ndarray
    attack_dynamic: np.ndarray


@lru_cache(maxsize=None)
def _default_numeric_catalog(card_count: int) -> NumericFeatureCatalog:
    from cg.api import all_attack, all_card_data

    cards = all_card_data()
    card_features = (
        build_card_feature_table(cards, card_count)
        .detach()
        .cpu()
        .numpy()
    )
    attacks = all_attack()
    attack_count = max((int(attack.attackId) for attack in attacks), default=-1) + 1
    attack_damage = np.zeros(attack_count, dtype=np.float32)
    for attack in attacks:
        attack_damage[int(attack.attackId)] = float(attack.damage) / 300.0
    card_attacks: list[tuple[int, ...]] = [()] * card_count
    for card in cards:
        card_id = int(card.cardId)
        if 0 <= card_id < card_count:
            card_attacks[card_id] = tuple(int(value) for value in card.attacks)
    return NumericFeatureCatalog(
        card_features=card_features,
        attack_damage=attack_damage,
        card_attacks=tuple(card_attacks),
    )


def enumerate_actions(
    option_count: int,
    min_count: int,
    max_count: int,
    limit: int = 64,
) -> list[list[int]]:
    """Enumerate unordered legal selections, preferring the notebook's max count."""
    if not 0 <= min_count <= max_count <= option_count:
        raise ValueError(
            "action counts must satisfy "
            f"0 <= min_count <= max_count <= option_count; got "
            f"{min_count}, {max_count}, {option_count}"
        )
    if limit < 1:
        raise ValueError("limit must be >= 1")
    actions: list[list[int]] = []
    for count in range(max_count, min_count - 1, -1):
        for selection in combinations(range(option_count), count):
            actions.append(list(selection))
            if len(actions) == limit:
                return actions
    return actions


def _add_card(sv: SparseVector, card: Any, card_count: int) -> None:
    if card is not None: sv.add(card.id, 1)
    sv.add_pos(card_count)


def _add_cards(sv: SparseVector, cards: Any, weight: float, card_count: int) -> None:
    if cards is not None:
        for card in cards: sv.add(card.id, weight)
    sv.add_pos(card_count)


def _add_pokemon(sv: SparseVector, poke: Any, card_count: int) -> None:
    if poke is None:
        sv.add_single(1); sv.add_pos(1 + 3 * card_count); return
    sv.add_single(0); sv.add_single(poke.hp / 400)
    _add_card(sv, poke, card_count); _add_cards(sv, poke.tools, 1, card_count)
    _add_cards(sv, poke.energyCards, .5, card_count)


def _active(player: Any) -> Any | None:
    return player.active[0] if player.active else None


def _one_hot(index: int, size: int, name: str) -> list[float]:
    index = int(index)
    if not 0 <= index < size:
        raise ValueError(f"{name}={index} is outside [0, {size})")
    result = [0.0] * size
    result[index] = 1.0
    return result


def _card_id(card: Any) -> int:
    return int(card.id)


def _player_summary(
    player: Any,
    catalog: NumericFeatureCatalog,
) -> list[float]:
    active = _active(player)
    bench = list(player.bench[:8])
    bench_energy = sum(len(pokemon.energyCards) for pokemon in bench)
    bench_hp = sum(float(pokemon.hp) for pokemon in bench)
    bench_max_hp = sum(float(pokemon.maxHp) for pokemon in bench)

    retreat_cost = 0.0
    max_attack_damage = 0.0
    attack_count = 0.0
    weakness_norm = 0.0
    resistance_norm = 0.0
    if active is not None:
        card_id = _card_id(active)
        if 0 <= card_id < len(catalog.card_features):
            card_features = catalog.card_features[card_id]
            retreat_cost = float(card_features[CARD_RETREAT_INDEX])
            weakness_norm = float(
                np.argmax(
                    card_features[
                        CARD_WEAKNESS_OFFSET:
                        CARD_WEAKNESS_OFFSET + CARD_WEAKNESS_DIM
                    ]
                )
            ) / 12.0
            resistance_norm = float(
                np.argmax(
                    card_features[
                        CARD_RESISTANCE_OFFSET:
                        CARD_RESISTANCE_OFFSET + CARD_RESISTANCE_DIM
                    ]
                )
            ) / 12.0
            attacks = catalog.card_attacks[card_id]
            attack_count = len(attacks) / 4.0
            max_attack_damage = max(
                (
                    float(catalog.attack_damage[attack_id])
                    for attack_id in attacks
                    if 0 <= attack_id < len(catalog.attack_damage)
                ),
                default=0.0,
            )

    features = [
        float(player.deckCount) / 60.0,
        float(player.handCount) / 20.0,
        len(player.discard) / 60.0,
    ]
    features.extend(_one_hot(len(player.prize), 7, "prize_count"))
    features.extend(
        [
            len(bench) / 8.0,
            float(active is not None),
            float(bool(player.poisoned)),
            float(bool(player.burned)),
            float(bool(player.asleep)),
            float(bool(player.paralyzed)),
            float(bool(player.confused)),
            (float(active.hp) / 400.0) if active is not None else 0.0,
            (float(active.maxHp) / 400.0) if active is not None else 0.0,
            (len(active.energyCards) / 10.0) if active is not None else 0.0,
            (len(active.tools) / 4.0) if active is not None else 0.0,
            (len(active.preEvolution) / 2.0) if active is not None else 0.0,
            retreat_cost,
            max_attack_damage,
            attack_count,
            weakness_norm,
            resistance_norm,
        ]
    )
    for slot in range(8):
        if slot < len(bench):
            pokemon = bench[slot]
            features.extend(
                [
                    1.0,
                    float(pokemon.hp) / 400.0,
                    len(pokemon.energyCards) / 5.0,
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0])
    features.extend(
        [
            bench_energy / 32.0,
            bench_hp / 3200.0,
            bench_max_hp / 3200.0,
        ]
    )
    if len(features) != 54:
        raise RuntimeError(f"player summary has {len(features)} dimensions")
    return features


def _visible_own_cards(player: Any) -> Counter[int]:
    visible: Counter[int] = Counter()

    def add_card(card: Any) -> None:
        if card is not None:
            visible[_card_id(card)] += 1

    def add_pokemon(pokemon: Any | None) -> None:
        if pokemon is None:
            return
        add_card(pokemon)
        for attached in (
            pokemon.energyCards,
            pokemon.tools,
            pokemon.preEvolution,
        ):
            for card in attached:
                add_card(card)

    for card in player.hand or []:
        add_card(card)
    for card in player.discard:
        add_card(card)
    add_pokemon(_active(player))
    for pokemon in player.bench:
        add_pokemon(pokemon)
    return visible


def _deck_remaining_summary(
    deck: list[int],
    own_player: Any,
    catalog: NumericFeatureCatalog,
) -> list[float]:
    remaining = Counter(int(card_id) for card_id in deck)
    remaining.subtract(_visible_own_cards(own_player))
    remaining = Counter(
        {
            card_id: max(0, count)
            for card_id, count in remaining.items()
        }
    )
    total = float(sum(remaining.values()))
    type_counts = [0.0] * CARD_TYPE_DIM
    stage_counts = [0.0, 0.0, 0.0]
    has_ex = 0.0
    has_mega = 0.0
    for card_id, count in remaining.items():
        if count <= 0 or not 0 <= card_id < len(catalog.card_features):
            continue
        card_features = catalog.card_features[card_id]
        card_type = int(
            np.argmax(
                card_features[
                    CARD_TYPE_OFFSET:CARD_TYPE_OFFSET + CARD_TYPE_DIM
                ]
            )
        )
        type_counts[card_type] += float(count)
        for stage in range(3):
            if card_features[CARD_STAGE_OFFSET + stage] > 0.5:
                stage_counts[stage] += float(count)
        has_ex = max(has_ex, float(card_features[CARD_SPECIAL_OFFSET] > 0.5))
        has_mega = max(
            has_mega,
            float(card_features[CARD_SPECIAL_OFFSET + 1] > 0.5),
        )
    pokemon = type_counts[0]
    energy = type_counts[5] + type_counts[6]
    result = (
        [value / 4.0 for value in type_counts]
        + [value / 4.0 for value in stage_counts]
        + [has_ex, has_mega]
        + [
            total / 60.0,
            pokemon / max(total, 1.0),
            energy / max(total, 1.0),
        ]
    )
    if len(result) != 15:
        raise RuntimeError("deck remaining summary must contain 15 values")
    return result


def _opponent_revealed_summary(
    player: Any,
    catalog: NumericFeatureCatalog,
) -> list[float]:
    revealed: list[Any] = list(player.discard)

    def add_pokemon(pokemon: Any | None) -> None:
        if pokemon is None:
            return
        revealed.append(pokemon)
        revealed.extend(pokemon.energyCards)
        revealed.extend(pokemon.tools)
        revealed.extend(pokemon.preEvolution)

    add_pokemon(_active(player))
    for pokemon in player.bench:
        add_pokemon(pokemon)

    card_ids = [
        _card_id(card)
        for card in revealed
        if 0 <= _card_id(card) < len(catalog.card_features)
    ]
    type_counts = [0.0] * CARD_TYPE_DIM
    has_ex = has_mega = has_tera = 0.0
    pokemon_count = 0
    average_hp = 0.0
    average_stage = 0.0
    for card_id in card_ids:
        card_features = catalog.card_features[card_id]
        card_type = int(
            np.argmax(
                card_features[
                    CARD_TYPE_OFFSET:CARD_TYPE_OFFSET + CARD_TYPE_DIM
                ]
            )
        )
        type_counts[card_type] += 1.0
        if card_type == 0:
            pokemon_count += 1
            average_hp += float(card_features[CARD_HP_INDEX])
            has_ex = max(has_ex, float(card_features[CARD_SPECIAL_OFFSET] > 0.5))
            has_mega = max(
                has_mega,
                float(card_features[CARD_SPECIAL_OFFSET + 1] > 0.5),
            )
            has_tera = max(
                has_tera,
                float(card_features[CARD_SPECIAL_OFFSET + 2] > 0.5),
            )
            average_stage += (
                float(card_features[CARD_STAGE_OFFSET + 1])
                + 2.0 * float(card_features[CARD_STAGE_OFFSET + 2])
            )
    if pokemon_count:
        average_hp /= pokemon_count
        average_stage /= pokemon_count
    energy_in_play = sum(
        len(pokemon.energyCards)
        for pokemon in ([_active(player)] + list(player.bench))
        if pokemon is not None
    )
    result = (
        [value / 10.0 for value in type_counts]
        + [has_ex, has_mega, has_tera]
        + [
            pokemon_count / 5.0,
            len(card_ids) / 20.0,
            average_hp,
            average_stage / 2.0,
            len(player.discard) / 20.0,
            len(player.bench) / 8.0,
            energy_in_play / 10.0,
        ]
    )
    if len(result) != 17:
        raise RuntimeError("opponent revealed summary must contain 17 values")
    return result


def _global_summary(obs: Any, yours: int) -> list[float]:
    state = obs.current
    select = obs.select
    first_relative = (
        -1.0
        if int(state.firstPlayer) < 0
        else float(int(state.firstPlayer) == yours)
    )
    result = [
        float(state.turn) / 100.0,
        float(state.turnActionCount) / 100.0,
        first_relative,
        float(bool(state.supporterPlayed)),
        float(bool(state.stadiumPlayed)),
        float(bool(state.energyAttached)),
        float(bool(state.retreated)),
        float(yours),
    ]
    result.extend(_one_hot(int(select.type), SELECT_TYPE_DIM, "select.type"))
    result.extend(
        _one_hot(
            int(select.context),
            SELECT_CONTEXT_DIM,
            "select.context",
        )
    )
    result.extend(
        [
            int(select.minCount) / 6.0,
            int(select.maxCount) / 6.0,
            int(select.remainDamageCounter) / 20.0,
            int(select.remainEnergyCost) / 10.0,
            len(select.option) / 64.0,
        ]
    )
    if len(result) != GLOBAL_SUMMARY_DIM:
        raise RuntimeError("global summary must contain 73 values")
    return result


def encoder_features(
    obs: Any,
    deck: list[int],
    card_count: int,
    *,
    numeric_catalog: NumericFeatureCatalog | None = None,
) -> EncoderFeatures:
    catalog = numeric_catalog or _default_numeric_catalog(card_count)
    state, yours, sparse = obs.current, obs.current.yourIndex, SparseVector()
    relative_players = [
        state.players[yours],
        state.players[1 - yours],
    ]
    pokemon_appear = []

    for player in relative_players:
        for slot in range(8):
            pokemon = (
                player.bench[slot]
                if slot < len(player.bench)
                else None
            )
            pokemon_appear.append(
                0 if pokemon is None
                else 2 if bool(pokemon.appearThisTurn)
                else 1
            )
            sparse.word_start()
            position = sparse.pos
            _add_pokemon(
                sparse,
                pokemon,
                card_count,
            )
            if slot != 7:
                sparse.pos = position
    for player in relative_players:
        pokemon = _active(player)
        pokemon_appear.append(
            0 if pokemon is None
            else 2 if bool(pokemon.appearThisTurn)
            else 1
        )
        sparse.word_start()
        _add_pokemon(sparse, pokemon, card_count)

    # Dense own and opponent summary placeholders.
    sparse.word_start()
    sparse.word_start()

    sparse.word_start()
    _add_cards(sparse, relative_players[0].discard, 0.25, card_count)
    sparse.word_start()
    _add_cards(sparse, relative_players[1].discard, 0.25, card_count)
    sparse.word_start()
    _add_cards(sparse, relative_players[0].hand, 0.25, card_count)
    sparse.word_start()
    for card_id in deck:
        sparse.add(card_id, 0.25)
    sparse.add_pos(card_count)
    sparse.word_start()
    _add_cards(sparse, state.stadium, 1.0, card_count)

    # Dense global summary placeholder.
    sparse.word_start()
    if len(sparse.offset) != ENCODER_TOKENS:
        raise RuntimeError(
            f"encoder produced {len(sparse.offset)} tokens"
        )
    if len(pokemon_appear) != POKEMON_ENCODER_TOKENS:
        raise RuntimeError(
            "encoder Pokemon appear state must contain 18 values"
        )

    own_summary = _player_summary(relative_players[0], catalog)
    own_summary.extend(
        _deck_remaining_summary(deck, relative_players[0], catalog)
    )
    opponent_summary = _player_summary(relative_players[1], catalog)
    opponent_summary.extend(
        _opponent_revealed_summary(relative_players[1], catalog)
    )
    if len(own_summary) != OWN_SUMMARY_DIM:
        raise RuntimeError("own summary must contain 60 values")
    if len(opponent_summary) != OPPONENT_SUMMARY_DIM:
        raise RuntimeError("opponent summary must contain 62 values")
    return EncoderFeatures(
        sparse=sparse,
        pokemon_appear=pokemon_appear,
        own_summary=own_summary,
        opponent_summary=opponent_summary,
        global_summary=_global_summary(obs, yours),
    )


def _optional_int(value: Any, default: int = 0) -> int:
    return default if value is None else int(value)


def _option_value_index(value: Any, name: str) -> int:
    if value is None:
        return 0
    number = int(value)
    if number < 0:
        raise ValueError(f"{name}={number} must be non-negative")
    if number > OPTION_VALUE_MAX:
        return OPTION_VALUE_MAX + 2
    return number + 1


def _area_cards(
    obs: Any,
    area: Any,
    player_index: int,
) -> list[Any]:
    from cg.api import AreaType

    player = obs.current.players[player_index]
    mapping = {
        AreaType.DECK: obs.select.deck,
        AreaType.HAND: player.hand,
        AreaType.DISCARD: player.discard,
        AreaType.ACTIVE: player.active,
        AreaType.BENCH: player.bench,
        AreaType.PRIZE: player.prize,
        AreaType.STADIUM: obs.current.stadium,
        AreaType.LOOKING: obs.current.looking,
    }
    return list(mapping.get(area) or [])


def _area_card(
    obs: Any,
    area: Any,
    index: Any,
    player_index: int,
) -> Any | None:
    cards = _area_cards(obs, area, player_index)
    position = _optional_int(index, -1)
    return cards[position] if 0 <= position < len(cards) else None


def _valid_card_id(card: Any, card_count: int) -> int:
    if card is None:
        return card_count
    card_id = int(card.id)
    return card_id if 0 <= card_id < card_count else card_count


def _is_pokemon(value: Any) -> bool:
    return value is not None and all(
        hasattr(value, name)
        for name in ("hp", "maxHp", "energies", "energyCards", "tools")
    )


def _pokemon_dynamic_features(
    pokemon: Any | None,
    *,
    is_active: bool,
    is_own: bool,
) -> np.ndarray:
    features = np.zeros(POKEMON_DYNAMIC_WORD_DIM, dtype=np.float32)
    if not _is_pokemon(pokemon):
        return features
    hp = max(float(pokemon.hp), 0.0)
    max_hp = max(float(pokemon.maxHp), 0.0)
    tools = list(pokemon.tools or [])
    energy_cards = list(pokemon.energyCards or [])
    energies = list(pokemon.energies or [])
    features[:8] = [
        1.0,
        hp / 400.0,
        max_hp / 400.0,
        max(max_hp - hp, 0.0) / 400.0,
        hp / max_hp if max_hp > 0 else 0.0,
        len(tools) / 4.0,
        len(energy_cards) / 10.0,
        len(energies) / 10.0,
    ]
    for energy in energies:
        energy_type = int(energy)
        if 0 <= energy_type < ENERGY_TYPE_DIM:
            features[8 + energy_type] += 0.1
    features[20:] = [
        float(bool(getattr(pokemon, "appearThisTurn", False))),
        float(is_active),
        float(is_own),
    ]
    return features


def _option_pokemon_slots(
    obs: Any,
    option: Any,
) -> tuple[tuple[Any | None, bool, bool], tuple[Any | None, bool, bool]]:
    from cg.api import AreaType, OptionType

    yours = int(obs.current.yourIndex)
    opponent = 1 - yours
    missing = (None, False, False)
    option_type = option.type
    if option_type == OptionType.ATTACK:
        return (
            (_active(obs.current.players[yours]), True, True),
            (_active(obs.current.players[opponent]), True, False),
        )
    if option_type == OptionType.RETREAT:
        return ((_active(obs.current.players[yours]), True, True), missing)

    player_index = _optional_int(option.playerIndex, yours)
    player_index = max(0, min(player_index, len(obs.current.players) - 1))
    if option_type in {
        OptionType.ABILITY,
        OptionType.CARD,
        OptionType.DISCARD,
        OptionType.TOOL_CARD,
        OptionType.ENERGY_CARD,
        OptionType.ENERGY,
    }:
        pokemon = _area_card(obs, option.area, option.index, player_index)
        return (
            (
                pokemon if _is_pokemon(pokemon) else None,
                option.area == AreaType.ACTIVE,
                player_index == yours,
            ),
            missing,
        )
    if option_type in {OptionType.ATTACH, OptionType.EVOLVE}:
        pokemon = _area_card(
            obs, option.inPlayArea, option.inPlayIndex, yours
        )
        return (
            (
                pokemon if _is_pokemon(pokemon) else None,
                option.inPlayArea == AreaType.ACTIVE,
                True,
            ),
            missing,
        )
    return missing, missing


def _option_entity_ids(
    obs: Any,
    option: Any,
    card_count: int,
    attack_count: int,
) -> tuple[int, int, int]:
    from cg.api import AreaType, OptionType

    yours = int(obs.current.yourIndex)
    player_index = _optional_int(option.playerIndex, yours)
    player_index = max(0, min(player_index, len(obs.current.players) - 1))
    candidate = None
    target = None

    if option.type == OptionType.PLAY:
        candidate = _area_card(
            obs, AreaType.HAND, option.index, yours
        )
    elif option.type in {
        OptionType.CARD,
        OptionType.TOOL_CARD,
        OptionType.ENERGY_CARD,
        OptionType.ENERGY,
        OptionType.ABILITY,
        OptionType.DISCARD,
    }:
        candidate = _area_card(
            obs, option.area, option.index, player_index
        )
        if option.type == OptionType.TOOL_CARD and candidate is not None:
            tools = list(candidate.tools or [])
            tool_index = _optional_int(option.toolIndex, -1)
            candidate = (
                tools[tool_index]
                if 0 <= tool_index < len(tools)
                else None
            )
        elif option.type in {
            OptionType.ENERGY_CARD,
            OptionType.ENERGY,
        } and candidate is not None:
            energies = list(candidate.energyCards or [])
            energy_index = _optional_int(option.energyIndex, -1)
            candidate = (
                energies[energy_index]
                if 0 <= energy_index < len(energies)
                else None
            )
    elif option.type in {OptionType.ATTACH, OptionType.EVOLVE}:
        candidate = _area_card(
            obs, option.area, option.index, player_index
        )
        target = _area_card(
            obs, option.inPlayArea, option.inPlayIndex, yours
        )
    elif option.type == OptionType.RETREAT:
        active = list(obs.current.players[yours].active or [])
        target = active[0] if active else None

    candidate_id = _valid_card_id(candidate, card_count)
    if candidate_id == card_count:
        raw_card_id = _optional_int(option.cardId, 0)
        if 0 < raw_card_id < card_count:
            candidate_id = raw_card_id
    target_id = _valid_card_id(target, card_count)
    raw_attack_id = option.attackId
    attack_id = (
        int(raw_attack_id)
        if raw_attack_id is not None
        and 0 <= int(raw_attack_id) < attack_count
        else attack_count
    )
    return candidate_id, target_id, attack_id


def decoder_features(
    obs: Any,
    actions: list[list[int]],
    card_count: int,
    attack_count: int,
    *,
    numeric_catalog: NumericFeatureCatalog | None = None,
) -> OptionFeatures:
    """Encode raw options once and retain exact action membership."""
    from cg.api import OptionType

    catalog = numeric_catalog or _default_numeric_catalog(card_count)
    options = list(obs.select.option)
    context = int(obs.select.context)
    if not 0 <= context < SELECT_CONTEXT_DIM:
        raise ValueError(f"select context {context} is outside the vocabulary")

    categorical = np.empty(
        (len(options), OPTION_CATEGORICAL_DIM), dtype=np.int64
    )
    numeric = np.zeros((len(options), OPTION_NUMERIC_DIM), dtype=np.float32)
    pokemon_dynamic = np.zeros(
        (len(options), POKEMON_DYNAMIC_DIM), dtype=np.float32
    )
    attack_dynamic = np.zeros(
        (len(options), ATTACK_DYNAMIC_DIM), dtype=np.float32
    )
    yours = int(obs.current.yourIndex)

    for option_index, option in enumerate(options):
        option_type = int(option.type)
        if not 0 <= option_type < OPTION_TYPE_DIM:
            raise ValueError(
                f"option type {option_type} is outside the vocabulary"
            )
        candidate_id, target_id, attack_id = _option_entity_ids(
            obs, option, card_count, attack_count
        )
        player_relation = (
            0 if option.playerIndex is None
            else 1 if int(option.playerIndex) == yours
            else 2
        )
        area_index = 0 if option.area is None else int(option.area)
        in_play_area_index = (
            0 if option.inPlayArea is None
            else int(option.inPlayArea)
        )
        special_condition_index = (
            0 if option.specialConditionType is None
            else int(option.specialConditionType) + 1
        )
        if not 0 <= area_index < OPTION_AREA_DIM:
            raise ValueError(f"area index {area_index} is outside the vocabulary")
        if not 0 <= in_play_area_index < OPTION_AREA_DIM:
            raise ValueError(
                "in-play area index "
                f"{in_play_area_index} is outside the vocabulary"
            )
        if not 0 <= special_condition_index < OPTION_SPECIAL_CONDITION_DIM:
            raise ValueError(
                "special condition index "
                f"{special_condition_index} is outside the vocabulary"
            )
        categorical[option_index] = [
            option_type,
            context,
            candidate_id,
            target_id,
            attack_id,
            _option_value_index(option.number, "option number"),
            _option_value_index(option.count, "option count"),
            player_relation,
            area_index,
            in_play_area_index,
            special_condition_index,
        ]
        numeric[option_index] = [
            _optional_int(option.index) / 60.0,
            _optional_int(option.toolIndex) / 4.0,
            _optional_int(option.energyIndex) / 10.0,
            _optional_int(option.inPlayIndex) / 5.0,
            (option_index + 1) / max(1, len(options)),
        ]
        primary, secondary = _option_pokemon_slots(obs, option)
        primary_features = _pokemon_dynamic_features(
            primary[0], is_active=primary[1], is_own=primary[2]
        )
        secondary_features = _pokemon_dynamic_features(
            secondary[0], is_active=secondary[1], is_own=secondary[2]
        )
        pokemon_dynamic[option_index] = np.concatenate(
            (primary_features, secondary_features)
        )

        if (
            option.type == OptionType.ATTACK
            and attack_id < attack_count
            and primary_features[0] > 0
            and secondary_features[0] > 0
        ):
            target_hp = max(float(secondary[0].hp), 0.0)
            base_damage = (
                float(catalog.attack_damage[attack_id]) * 300.0
                if attack_id < len(catalog.attack_damage)
                else 0.0
            )
            super_effective = False
            resisted = False
            own_id = _valid_card_id(primary[0], card_count)
            opponent_id = _valid_card_id(secondary[0], card_count)
            if own_id < card_count and opponent_id < card_count:
                own_energy = catalog.card_features[
                    own_id,
                    CARD_ENERGY_TYPE_OFFSET:
                    CARD_ENERGY_TYPE_OFFSET + ENERGY_TYPE_DIM,
                ]
                if np.any(own_energy > 0.5):
                    own_energy_type = int(np.argmax(own_energy))
                    super_effective = bool(
                        catalog.card_features[
                            opponent_id,
                            CARD_WEAKNESS_OFFSET + own_energy_type,
                        ]
                        > 0.5
                    )
                    resisted = bool(
                        catalog.card_features[
                            opponent_id,
                            CARD_RESISTANCE_OFFSET + own_energy_type,
                        ]
                        > 0.5
                    )
            attack_dynamic[option_index] = [
                1.0,
                min(base_damage / max(target_hp, 1.0), 4.0) / 4.0,
                float(base_damage >= target_hp),
                max(target_hp - base_damage, 0.0) / 400.0,
                float(super_effective),
                float(resisted),
            ]

    action_index: list[int] = []
    action_offset = [0]
    for action in actions:
        if len(set(action)) != len(action) or any(
            index < 0 or index >= len(options) for index in action
        ):
            raise ValueError("action contains an invalid option index")
        action_index.extend(action)
        action_offset.append(len(action_index))
    return OptionFeatures(
        categorical=categorical,
        numeric=numeric,
        pokemon_dynamic=pokemon_dynamic,
        attack_dynamic=attack_dynamic,
        action_index=np.asarray(action_index, dtype=np.int64),
        action_offset=np.asarray(action_offset, dtype=np.int64),
    )


def _history_structural_option(obs: Any, option: Any) -> list[int]:
    """Return position-free structural fields for one historical option."""
    from cg.api import AreaType, OptionType

    yours = int(obs.current.yourIndex)
    option_type = int(option.type)
    source_area = 0 if option.area is None else int(option.area)
    target_area = 0 if option.inPlayArea is None else int(option.inPlayArea)
    source_relation = (
        0
        if option.playerIndex is None
        else 1 if int(option.playerIndex) == yours else 2
    )
    target_relation = 0

    if option.type == OptionType.PLAY:
        source_area = int(AreaType.HAND)
        source_relation = 1
    elif option.type == OptionType.ATTACK:
        source_area = int(AreaType.ACTIVE)
        target_area = int(AreaType.ACTIVE)
        source_relation = 1
        target_relation = 2
    elif option.type == OptionType.RETREAT:
        source_area = int(AreaType.ACTIVE)
        source_relation = 1
    elif option.type in {OptionType.ATTACH, OptionType.EVOLVE}:
        if source_relation == 0:
            source_relation = 1
        if target_area != 0:
            target_relation = 1

    if not 0 <= source_area < OPTION_AREA_DIM:
        raise ValueError(
            f"history source area {source_area} is outside the vocabulary"
        )
    if not 0 <= target_area < OPTION_AREA_DIM:
        raise ValueError(
            f"history target area {target_area} is outside the vocabulary"
        )
    return [
        option_type,
        source_area,
        target_area,
        source_relation,
        target_relation,
        _option_value_index(option.number, "history option number"),
        _option_value_index(option.count, "history option count"),
        (
            0
            if option.specialConditionType is None
            else int(option.specialConditionType) + 1
        ),
    ]


def history_action_features(
    obs: Any,
    selected: list[int],
    card_count: int,
    attack_count: int,
    *,
    numeric_catalog: NumericFeatureCatalog | None = None,
    encoded_options: OptionFeatures | None = None,
) -> HistoryActionFeatures:
    """Encode one completed action without any option-position numerics."""
    options = list(obs.select.option)
    chosen = [int(index) for index in selected]
    if len(set(chosen)) != len(chosen) or any(
        index < 0 or index >= len(options) for index in chosen
    ):
        raise ValueError("historical action contains an invalid option index")
    select_type = int(obs.select.type)
    if not 0 <= select_type < SELECT_TYPE_DIM:
        raise ValueError(f"select type {select_type} is outside the vocabulary")

    encoded = encoded_options
    if encoded is None:
        encoded = decoder_features(
            obs,
            [chosen],
            card_count,
            attack_count,
            numeric_catalog=numeric_catalog,
        )
    elif encoded.categorical.shape[0] != len(options):
        raise ValueError("encoded_options does not align with observation options")
    rows = np.asarray(chosen, dtype=np.int64)
    structural = np.asarray(
        [_history_structural_option(obs, options[index]) for index in chosen],
        dtype=np.int64,
    ).reshape(-1, HISTORY_STRUCTURAL_DIM)
    return HistoryActionFeatures(
        select_type=select_type,
        select_context=int(obs.select.context),
        option_categorical=encoded.categorical[rows].copy(),
        structural=structural,
        pokemon_dynamic=encoded.pokemon_dynamic[rows].copy(),
        attack_dynamic=encoded.attack_dynamic[rows].copy(),
    )
