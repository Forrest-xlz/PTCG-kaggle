"""Public API for deterministic validation-deck selection.

The implementation currently shares internal indexes and validation helpers
with :mod:`analysis.deck_statistics`; this module provides the focused domain
boundary used by selection notebooks and callers.
"""

from analysis.deck_statistics import (
    IsolationRoll,
    IsolationSamplingError,
    archetype_core_card_ids,
    audit_deck_archetype_selection,
    build_archetype_isolation_candidates,
    build_archetype_selection_details,
    build_deck_isolation_candidates,
    build_top_deck_archetype_candidates,
    roll_archetype_isolation_selection,
    roll_deck_isolation_selection,
    roll_top_deck_archetype_selection,
)

__all__ = [
    "IsolationRoll",
    "IsolationSamplingError",
    "archetype_core_card_ids",
    "audit_deck_archetype_selection",
    "build_archetype_isolation_candidates",
    "build_archetype_selection_details",
    "build_deck_isolation_candidates",
    "build_top_deck_archetype_candidates",
    "roll_archetype_isolation_selection",
    "roll_deck_isolation_selection",
    "roll_top_deck_archetype_selection",
]
