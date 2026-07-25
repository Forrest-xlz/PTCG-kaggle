"""Sparse feature construction adapted from the competition notebook.

The public functions accept the typed objects returned by
``cg.api.to_observation_class``.  Keeping this module separate makes feature
parity testable and prevents extraction from depending on PyTorch.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable


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


def _add_player(sv: SparseVector, ps: Any, card_count: int) -> None:
    for value in (ps.deckCount / 60, len(ps.discard) / 60, ps.handCount / 8, len(ps.bench) / 5): sv.add_single(value)
    sv.add(len(ps.prize), 1); sv.add_pos(7)
    for value in (ps.poisoned, ps.burned, ps.asleep, ps.paralyzed, ps.confused): sv.add_single(value)
    _add_cards(sv, ps.discard, .25, card_count)


def encoder_features(obs: Any, deck: list[int], card_count: int) -> SparseVector:
    state, yours, sv = obs.current, obs.current.yourIndex, SparseVector()
    for i in range(2):
        ps = state.players[i ^ yours]
        for j in range(8):
            sv.word_start(); pos = sv.pos
            _add_pokemon(sv, ps.bench[j] if j < len(ps.bench) else None, card_count)
            if j != 7: sv.pos = pos
    for i in range(2):
        ps = state.players[i ^ yours]; sv.word_start()
        _add_pokemon(sv, ps.active[0] if ps.active else None, card_count)
    for i in range(2):
        sv.word_start(); _add_player(sv, state.players[i ^ yours], card_count)
    sv.word_start(); _add_cards(sv, state.players[yours].hand, .25, card_count)
    sv.word_start()
    for card_id in deck: sv.add(card_id, .25)
    sv.add_pos(card_count)
    sv.word_start(); _add_cards(sv, state.stadium, 1, card_count)
    sv.word_start(); sv.add_single(1); sv.add_single(state.turn / 10); sv.add_single(state.firstPlayer == yours)
    return sv


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
