from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beam_search.config import DeckSpec
from beam_search.determinization import build_search_inputs


def card(card_id: int, player: int, *, serial: int | None = None):
    return SimpleNamespace(
        id=card_id,
        playerIndex=player,
        serial=card_id * 10 + player if serial is None else serial,
    )


def pokemon(
    card_id: int,
    player: int,
    *,
    tools=(),
    energies=(),
    previous=(),
):
    return SimpleNamespace(
        id=card_id,
        playerIndex=player,
        serial=card_id * 10 + player,
        tools=list(tools),
        energyCards=list(energies),
        preEvolution=list(previous),
    )


def observation():
    own = SimpleNamespace(
        hand=[card(0, 0), card(1, 0)],
        handCount=2,
        active=[
            pokemon(
                2,
                0,
                tools=[card(3, 0)],
                energies=[card(4, 0)],
                previous=[card(5, 0)],
            )
        ],
        bench=[pokemon(6, 0)],
        discard=[card(7, 0)],
        prize=[None, card(8, 0)],
        deckCount=50,
    )
    opponent = SimpleNamespace(
        hand=None,
        handCount=3,
        active=[pokemon(10, 1)],
        bench=[pokemon(11, 1)],
        discard=[card(12, 1)],
        prize=[None, card(13, 1)],
        deckCount=52,
    )
    return SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0,
            players=[own, opponent],
            stadium=[],
            looking=None,
        ),
        select=SimpleNamespace(
            deck=None,
            effect=None,
            contextCard=None,
        ),
    )


def test_hidden_assignments_preserve_counts_positions_and_seed() -> None:
    obs = observation()
    own = DeckSpec("own", tuple(range(60)))
    opponent = DeckSpec("opponent", tuple(range(60)))

    first = build_search_inputs(obs, own, opponent, random.Random(7))
    second = build_search_inputs(obs, own, opponent, random.Random(7))

    assert first == second
    assert len(first.your_deck) == 50
    assert len(first.your_prize) == 2
    assert first.your_prize[1] == 8
    assert len(first.opponent_deck) == 52
    assert len(first.opponent_prize) == 2
    assert first.opponent_prize[1] == 13
    assert len(first.opponent_hand) == 3
    assert first.opponent_active == ()
    assert first.warnings == ()

    own_visible = [0, 1, 2, 3, 4, 5, 6, 7]
    own_all = own_visible + list(first.your_deck) + list(first.your_prize)
    assert Counter(own_all) == Counter(own.cards)
    opponent_visible = [10, 11, 12]
    opponent_all = (
        opponent_visible
        + list(first.opponent_deck)
        + list(first.opponent_prize)
        + list(first.opponent_hand)
    )
    assert Counter(opponent_all) == Counter(opponent.cards)


def test_different_seed_changes_unknown_assignment() -> None:
    obs = observation()
    deck = DeckSpec("deck", tuple(range(60)))
    first = build_search_inputs(obs, deck, deck, random.Random(1))
    second = build_search_inputs(obs, deck, deck, random.Random(2))
    assert first.your_deck != second.your_deck


def test_missing_generated_visible_card_warns_without_corrupting_pool() -> None:
    obs = observation()
    obs.current.players[0].discard.append(card(999, 0))
    deck = DeckSpec("deck", tuple(range(60)))

    result = build_search_inputs(obs, deck, deck, random.Random(1))

    assert len(result.your_deck) == 50
    assert len(result.warnings) == 1
    assert "visible card 999" in result.warnings[0]


def test_inconsistent_hidden_zone_size_is_rejected() -> None:
    obs = observation()
    obs.current.players[0].deckCount = 49
    deck = DeckSpec("deck", tuple(range(60)))

    with pytest.raises(ValueError, match="hidden zone sizes"):
        build_search_inputs(obs, deck, deck, random.Random(1))


def test_effect_card_is_removed_from_hidden_pool() -> None:
    obs = observation()
    obs.select.effect = card(9, 0)
    obs.current.players[0].deckCount = 49
    deck = DeckSpec("deck", tuple(range(60)))

    result = build_search_inputs(obs, deck, deck, random.Random(1))

    assert len(result.your_deck) == 49
    assert 9 not in result.your_deck


def test_looking_cards_are_removed_from_hidden_pool() -> None:
    obs = observation()
    obs.current.looking = [card(card_id, 0) for card_id in range(9, 16)]
    obs.current.players[0].deckCount = 43
    deck = DeckSpec("deck", tuple(range(60)))

    result = build_search_inputs(obs, deck, deck, random.Random(1))

    assert len(result.your_deck) == 43
    assert not set(range(9, 16)).intersection(result.your_deck)


def test_effect_already_in_visible_zone_is_not_removed_twice() -> None:
    obs = observation()
    discard_card = obs.current.players[0].discard[0]
    obs.select.effect = card(7, 0, serial=discard_card.serial)
    deck = DeckSpec("deck", tuple(range(60)))

    result = build_search_inputs(obs, deck, deck, random.Random(1))

    assert len(result.your_deck) == 50


def test_context_card_does_not_change_deck_conservation() -> None:
    obs = observation()
    obs.select.contextCard = card(9, 0)
    deck = DeckSpec("deck", tuple(range(60)))

    result = build_search_inputs(obs, deck, deck, random.Random(1))

    assert len(result.your_deck) == 50
