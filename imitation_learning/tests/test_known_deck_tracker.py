from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = PROJECT_ROOT / "model" / "known_deck.py"
SPEC = importlib.util.spec_from_file_location("known_deck", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
known_deck = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(known_deck)
KnownDeckTracker = known_deck.KnownDeckTracker


DRAW = 4
MOVE_CARD = 6
SHUFFLE = 0
DECK = 1
HAND = 2
LOOKING = 12


def move(
    *,
    player: int = 0,
    serial: int | None = 101,
    card_id: int | None = 741,
    from_area: int = HAND,
    to_area: int = DECK,
) -> dict:
    return {
        "type": MOVE_CARD,
        "playerIndex": player,
        "serial": serial,
        "cardId": card_id,
        "fromArea": from_area,
        "toArea": to_area,
    }


def test_tracker_maintains_current_physical_cards_idempotently() -> None:
    tracker = KnownDeckTracker()

    tracker.update([move(), move()], player_index=0)
    assert tracker.card_ids() == (741,)

    tracker.update(
        [move(serial=202, card_id=741, from_area=LOOKING)],
        player_index=0,
    )
    assert tracker.card_ids() == (741, 741)

    tracker.update(
        [{"type": SHUFFLE, "playerIndex": 0}], player_index=0
    )
    assert tracker.card_ids() == (741, 741)

    tracker.update(
        [{"type": DRAW, "playerIndex": 0, "serial": 101, "cardId": 741}],
        player_index=0,
    )
    assert tracker.card_ids() == (741,)

    tracker.update(
        [move(serial=202, from_area=DECK, to_area=HAND)],
        player_index=0,
    )
    assert tracker.card_ids() == ()


def test_tracker_ignores_opponent_and_malformed_events_and_resets() -> None:
    tracker = KnownDeckTracker()
    tracker.update(
        [
            move(player=1),
            move(serial=None),
            move(card_id=None),
        ],
        player_index=0,
    )
    assert tracker.card_ids() == ()

    tracker.update(
        [
            SimpleNamespace(
                type=MOVE_CARD,
                playerIndex=0,
                serial=303,
                cardId=1122,
                fromArea=HAND,
                toArea=DECK,
            )
        ],
        player_index=0,
    )
    assert tracker.card_ids() == (1122,)
    tracker.reset()
    assert tracker.card_ids() == ()
