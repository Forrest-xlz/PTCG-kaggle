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
    build_card_feature_table,
)


ENCODER_TOKENS = 20
OWN_SUMMARY_DIM = 60
OPPONENT_SUMMARY_DIM = 62
GLOBAL_SUMMARY_DIM = 73
SELECT_TYPE_DIM = 11
SELECT_CONTEXT_DIM = 49


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
    own_summary: list[float]
    opponent_summary: list[float]
    global_summary: list[float]


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
    bench = list(player.bench[:5])
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
            len(bench) / 5.0,
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
    for slot in range(5):
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
            bench_energy / 20.0,
            bench_hp / 2000.0,
            bench_max_hp / 2000.0,
        ]
    )
    if len(features) != 45:
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
            len(player.bench) / 5.0,
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

    for player in relative_players:
        for slot in range(5):
            sparse.word_start()
            position = sparse.pos
            _add_pokemon(
                sparse,
                player.bench[slot] if slot < len(player.bench) else None,
                card_count,
            )
            if slot != 4:
                sparse.pos = position
    for player in relative_players:
        sparse.word_start()
        _add_pokemon(sparse, _active(player), card_count)

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
        own_summary=own_summary,
        opponent_summary=opponent_summary,
        global_summary=_global_summary(obs, yours),
    )


def decoder_features(obs: Any, actions: list[list[int]], card_count: int, attack_count: int) -> SparseVector:
    """Build exact decoder features; enum integer values follow the cg API."""
    from cg.api import AreaType, OptionType
    sv, yours, ps, ctx = SparseVector(), obs.current.yourIndex, obs.current.players[obs.current.yourIndex], obs.select.context
    card_offset = 14 + attack_count
    def get_card(o, area=None, index=None, player_index=None):
        area = o.area if area is None else area
        index = o.index if index is None else index
        # The notebook uses the current player for ordinary actions such as
        # ATTACH/EVOLVE/ABILITY/DISCARD.  playerIndex is only supplied for
        # selection options that may explicitly point at either player.
        player_index = yours if player_index is None else player_index
        player = obs.current.players[player_index]
        mapping = {AreaType.DECK: obs.select.deck, AreaType.HAND: player.hand, AreaType.DISCARD: player.discard,
                   AreaType.ACTIVE: player.active, AreaType.BENCH: player.bench, AreaType.PRIZE: player.prize,
                   AreaType.STADIUM: obs.current.stadium, AreaType.LOOKING: obs.current.looking}
        cards = mapping.get(area); return cards[index] if cards is not None else None
    def main_feature(n, card):
        if card is not None: sv.add(card_offset + n * card_count + card.id, 1)
    def context_card(card):
        if card is not None: sv.add(card_offset + (8 + int(ctx)) * card_count + card.id, 1)
    for action in actions:
        sv.word_start()
        if not action: sv.add(0, 1); continue
        for idx in action:
            o = obs.select.option[idx]
            if o.type == OptionType.END: sv.add(1, 1)
            elif o.type == OptionType.YES: sv.add(2, 1)
            elif o.type == OptionType.NO: sv.add(3, 1)
            elif o.type == OptionType.SPECIAL_CONDITION: sv.add(4 + o.specialConditionType, 1)
            elif o.type == OptionType.NUMBER: sv.add(9 + min(o.number, 4), 1)
            elif o.type == OptionType.ATTACK: sv.add(14 + o.attackId, 1)
            elif o.type == OptionType.PLAY: main_feature(0, ps.hand[o.index])
            elif o.type == OptionType.ATTACH: main_feature(1, get_card(o)); main_feature(2, get_card(o, o.inPlayArea, o.inPlayIndex))
            elif o.type == OptionType.EVOLVE: main_feature(3, get_card(o)); main_feature(4, get_card(o, o.inPlayArea, o.inPlayIndex))
            elif o.type == OptionType.ABILITY: main_feature(5, get_card(o))
            elif o.type == OptionType.DISCARD: main_feature(6, get_card(o))
            elif o.type == OptionType.RETREAT: main_feature(7, ps.active[0])
            elif o.type == OptionType.CARD: context_card(get_card(o, player_index=o.playerIndex))
            elif o.type == OptionType.TOOL_CARD: context_card(get_card(o, player_index=o.playerIndex).tools[o.toolIndex])
            elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY): context_card(get_card(o, player_index=o.playerIndex).energyCards[o.energyIndex])
            elif o.type == OptionType.SKILL: sv.add(card_offset + (8 + int(ctx)) * card_count + o.cardId, 1)
    return sv
