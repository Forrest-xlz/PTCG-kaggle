"""Track publicly known cards that remain in either player's hand."""
from __future__ import annotations

from typing import Any, Iterable


HAND_AREA = 2
MOVE_CARD_LOG = 6
MOVE_CARD_REVERSE_LOG = 7
HAND_DEPARTURE_LOGS = frozenset({10, 11, 12})  # PLAY, ATTACH, EVOLVE


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class RevealedHandTracker:
    """Maintain exact public hand knowledge by physical card serial."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._cards: tuple[dict[int, int], dict[int, int]] = ({}, {})

    def update(self, logs: Iterable[Any] | None) -> None:
        for log in logs or ():
            log_type = _optional_int(getattr(log, "type", None))
            player = _optional_int(getattr(log, "playerIndex", None))
            if player not in (0, 1) or log_type is None:
                continue
            known = self._cards[player]
            serial = _optional_int(getattr(log, "serial", None))

            if log_type == MOVE_CARD_LOG:
                source = _optional_int(getattr(log, "fromArea", None))
                target = _optional_int(getattr(log, "toArea", None))
                if source == HAND_AREA:
                    if serial is None:
                        known.clear()
                    else:
                        known.pop(serial, None)
                if target == HAND_AREA:
                    card_id = _optional_int(getattr(log, "cardId", None))
                    if serial is not None and card_id is not None and card_id >= 0:
                        known[serial] = card_id
                continue

            if log_type == MOVE_CARD_REVERSE_LOG:
                if _optional_int(getattr(log, "fromArea", None)) == HAND_AREA:
                    known.clear()
                continue

            if log_type in HAND_DEPARTURE_LOGS:
                if serial is None:
                    known.clear()
                else:
                    known.pop(serial, None)

    def relative_cards(self, your_index: int) -> tuple[list[int], list[int]]:
        yours = int(your_index)
        if yours not in (0, 1):
            raise ValueError("your_index must be 0 or 1")

        def ordered(player: int) -> list[int]:
            return [
                card_id
                for _, card_id in sorted(self._cards[player].items())
            ]

        return ordered(yours), ordered(1 - yours)
