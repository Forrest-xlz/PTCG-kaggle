"""Track physical cards currently known to be inside the acting player's deck."""
from __future__ import annotations

from typing import Any, Iterable


LOG_SHUFFLE = 0
LOG_DRAW = 4
LOG_MOVE_CARD = 6
AREA_DECK = 1


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


class KnownDeckTracker:
    """Maintain an idempotent serial-to-Card-ID view of known deck contents."""

    def __init__(self) -> None:
        self._cards: dict[int, int] = {}

    def reset(self) -> None:
        self._cards.clear()

    def update(self, logs: Iterable[Any], player_index: int) -> None:
        player_index = int(player_index)
        for log in logs or ():
            if _integer(_field(log, "playerIndex")) != player_index:
                continue
            log_type = _integer(_field(log, "type"))
            serial = _integer(_field(log, "serial"))
            if serial is None:
                continue
            if log_type == LOG_DRAW:
                self._cards.pop(serial, None)
                continue
            if log_type != LOG_MOVE_CARD:
                continue
            from_area = _integer(_field(log, "fromArea"))
            to_area = _integer(_field(log, "toArea"))
            if from_area == AREA_DECK:
                self._cards.pop(serial, None)
            if to_area == AREA_DECK:
                card_id = _integer(_field(log, "cardId"))
                if card_id is not None and card_id >= 0:
                    self._cards[serial] = card_id

    def card_ids(self) -> tuple[int, ...]:
        return tuple(self._cards[serial] for serial in sorted(self._cards))
