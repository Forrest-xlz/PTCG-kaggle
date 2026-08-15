"""Reproducible hidden-card assignments for the CG Search API."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from beam_search.config import DeckSpec


@dataclass(frozen=True, slots=True)
class SearchInputs:
    your_deck: tuple[int, ...]
    your_prize: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    opponent_active: tuple[int, ...]
    warnings: tuple[str, ...]


def _card_ids(cards: Iterable[Any] | None) -> list[int]:
    return [int(card.id) for card in (cards or []) if card is not None]


def _pokemon_ids(pokemon: Any | None) -> list[int]:
    if pokemon is None:
        return []
    result = [int(pokemon.id)]
    result.extend(_card_ids(getattr(pokemon, "tools", None)))
    result.extend(_card_ids(getattr(pokemon, "energyCards", None)))
    result.extend(_card_ids(getattr(pokemon, "preEvolution", None)))
    return result


def _visible_player_cards(state: Any, player_index: int) -> list[int]:
    player = state.players[player_index]
    result = _card_ids(player.hand)
    result.extend(_card_ids(player.discard))
    result.extend(_card_ids(player.prize))
    for pokemon in list(player.active or []) + list(player.bench or []):
        result.extend(_pokemon_ids(pokemon))
    for stadium in list(state.stadium or []):
        if int(getattr(stadium, "playerIndex", -1)) == player_index:
            result.append(int(stadium.id))
    return result


def _remaining_pool(
    deck: DeckSpec,
    visible: Iterable[int],
    player_index: int,
    warnings: list[str],
) -> list[int]:
    counts = Counter(deck.cards)
    for card_id in visible:
        if counts[card_id] > 0:
            counts[card_id] -= 1
        else:
            warnings.append(
                f"player {player_index} visible card {card_id} is not "
                f"available in configured deck {deck.name!r}"
            )
    result: list[int] = []
    for card_id in deck.cards:
        if counts[card_id] > 0:
            result.append(card_id)
            counts[card_id] -= 1
    return result


def _fill_prize(
    visible_prize: Iterable[Any],
    pool: list[int],
    cursor: int,
) -> tuple[tuple[int, ...], int]:
    result: list[int] = []
    for card in visible_prize:
        if card is None:
            result.append(pool[cursor])
            cursor += 1
        else:
            result.append(int(card.id))
    return tuple(result), cursor


def _hidden_active_index(
    pool: list[int], basic_card_ids: frozenset[int]
) -> int:
    if basic_card_ids:
        for index, card_id in enumerate(pool):
            if card_id in basic_card_ids:
                return index
        raise ValueError("no Basic Pokemon is available for hidden Active")
    return 0


def build_search_inputs(
    obs: Any,
    your_deck: DeckSpec,
    opponent_deck: DeckSpec,
    rng: random.Random,
    *,
    basic_card_ids: frozenset[int] = frozenset(),
) -> SearchInputs:
    state = obs.current
    yours = int(state.yourIndex)
    opponent = 1 - yours
    warnings: list[str] = []
    your_player = state.players[yours]
    opponent_player = state.players[opponent]
    your_pool = _remaining_pool(
        your_deck,
        _visible_player_cards(state, yours),
        yours,
        warnings,
    )
    opponent_pool = _remaining_pool(
        opponent_deck,
        _visible_player_cards(state, opponent),
        opponent,
        warnings,
    )

    hidden_opponent_active = bool(
        opponent_player.active and opponent_player.active[0] is None
    )
    your_required = int(your_player.deckCount) + sum(
        card is None for card in your_player.prize
    )
    opponent_required = (
        int(opponent_player.deckCount)
        + int(opponent_player.handCount)
        + sum(card is None for card in opponent_player.prize)
        + int(hidden_opponent_active)
    )
    if len(your_pool) != your_required:
        raise ValueError(
            "your hidden zone sizes do not match the configured deck: "
            f"pool={len(your_pool)} required={your_required}; "
            + "; ".join(warnings)
        )
    if len(opponent_pool) != opponent_required:
        raise ValueError(
            "opponent hidden zone sizes do not match the configured deck: "
            f"pool={len(opponent_pool)} required={opponent_required}; "
            + "; ".join(warnings)
        )

    rng.shuffle(your_pool)
    rng.shuffle(opponent_pool)
    opponent_active: tuple[int, ...] = ()
    if hidden_opponent_active:
        index = _hidden_active_index(opponent_pool, basic_card_ids)
        opponent_active = (opponent_pool.pop(index),)

    cursor = 0
    your_prize, cursor = _fill_prize(your_player.prize, your_pool, cursor)
    sampled_your_deck = tuple(
        your_pool[cursor : cursor + int(your_player.deckCount)]
    )
    cursor = 0
    opponent_prize, cursor = _fill_prize(
        opponent_player.prize, opponent_pool, cursor
    )
    opponent_hand = tuple(
        opponent_pool[cursor : cursor + int(opponent_player.handCount)]
    )
    cursor += int(opponent_player.handCount)
    sampled_opponent_deck = tuple(
        opponent_pool[cursor : cursor + int(opponent_player.deckCount)]
    )
    return SearchInputs(
        your_deck=sampled_your_deck,
        your_prize=your_prize,
        opponent_deck=sampled_opponent_deck,
        opponent_prize=opponent_prize,
        opponent_hand=opponent_hand,
        opponent_active=opponent_active,
        warnings=tuple(warnings),
    )
