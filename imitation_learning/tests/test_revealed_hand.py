from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.revealed_hand import RevealedHandTracker


def _log(
    log_type: int,
    *,
    player: int = 0,
    card: int | None = None,
    serial: int | None = None,
    source: int | None = None,
    target: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        type=log_type,
        playerIndex=player,
        cardId=card,
        serial=serial,
        fromArea=source,
        toArea=target,
    )


class RevealedHandTrackerTests(unittest.TestCase):
    def test_public_cards_enter_and_leave_known_hand(self) -> None:
        tracker = RevealedHandTracker()
        tracker.update([
            _log(6, card=741, serial=10, source=12, target=2)
        ])
        self.assertEqual(tracker.relative_cards(0), ([741], []))

        tracker.update([_log(10, card=741, serial=10)])
        self.assertEqual(tracker.relative_cards(0), ([], []))

    def test_duplicate_card_ids_are_preserved_by_serial(self) -> None:
        tracker = RevealedHandTracker()
        tracker.update(
            [
                _log(6, card=741, serial=10, source=12, target=2),
                _log(6, card=741, serial=11, source=12, target=2),
            ]
        )
        self.assertEqual(tracker.relative_cards(0), ([741, 741], []))

    def test_private_draw_does_not_reveal_card(self) -> None:
        tracker = RevealedHandTracker()
        tracker.update([_log(4, card=741, serial=10)])
        self.assertEqual(tracker.relative_cards(0), ([], []))

    def test_face_down_hand_departure_clears_player_knowledge(self) -> None:
        tracker = RevealedHandTracker()
        tracker.update(
            [_log(6, player=1, card=305, serial=20, source=12, target=2)]
        )
        tracker.update([_log(7, player=1, source=2, target=1)])
        self.assertEqual(tracker.relative_cards(0), ([], []))

    def test_relative_cards_follow_current_player_perspective(self) -> None:
        tracker = RevealedHandTracker()
        tracker.update(
            [
                _log(6, player=0, card=741, serial=10, source=12, target=2),
                _log(6, player=1, card=305, serial=20, source=12, target=2),
            ]
        )
        self.assertEqual(tracker.relative_cards(0), ([741], [305]))
        self.assertEqual(tracker.relative_cards(1), ([305], [741]))

    def test_reset_discards_previous_match_state(self) -> None:
        tracker = RevealedHandTracker()
        tracker.update(
            [_log(6, card=741, serial=10, source=12, target=2)]
        )
        tracker.reset()
        self.assertEqual(tracker.relative_cards(0), ([], []))


if __name__ == "__main__":
    unittest.main()
