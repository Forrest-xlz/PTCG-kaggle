"""Shared parsing and display names for configured top decks."""
from __future__ import annotations


def parse_top_decks(raw):
    if not isinstance(raw, list) or not raw:
        raise ValueError("top_decks must be a non-empty list")
    decks, names = [], []
    for index, entry in enumerate(raw, 1):
        if isinstance(entry, dict):
            name = entry.get("name")
            deck = entry.get("card_ids")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each top deck requires a non-empty name")
            name = name.strip()
            if any(c in name for c in '/\\()\n\r'):
                raise ValueError("Top deck names must not contain slashes, parentheses or newlines")
        else:
            name, deck = str(index), entry
        if not isinstance(deck, list) or len(deck) != 60 or any(type(c) is not int or c < 0 for c in deck):
            raise ValueError(f"top_decks[{index}] requires 60 non-negative integer card IDs")
        if name in names:
            raise ValueError(f"Duplicate top deck name: {name}")
        if isinstance(entry, dict) and sorted(deck) in [sorted(d) for d in decks]:
            raise ValueError("top_decks contains duplicate exact decks")
        names.append(name)
        decks.append(deck)
    return decks, tuple(names)


def deck_label(index, names=()):
    return f"deck({names[index - 1]})" if names else f"deck{index}"
