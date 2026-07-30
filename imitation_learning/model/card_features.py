"""Static card features shared by training and inference."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


CARD_TYPE_DIM = 7
ENERGY_TYPE_DIM = 12
CARD_TYPE_OFFSET = 0
CARD_ENERGY_TYPE_OFFSET = CARD_TYPE_OFFSET + CARD_TYPE_DIM
CARD_HP_INDEX = CARD_ENERGY_TYPE_OFFSET + ENERGY_TYPE_DIM
CARD_RETREAT_INDEX = CARD_HP_INDEX + 1
CARD_WEAKNESS_OFFSET = CARD_RETREAT_INDEX + 1
CARD_WEAKNESS_DIM = ENERGY_TYPE_DIM + 1
CARD_RESISTANCE_OFFSET = CARD_WEAKNESS_OFFSET + CARD_WEAKNESS_DIM
CARD_RESISTANCE_DIM = ENERGY_TYPE_DIM + 1
CARD_STAGE_OFFSET = CARD_RESISTANCE_OFFSET + CARD_RESISTANCE_DIM
CARD_SPECIAL_OFFSET = CARD_STAGE_OFFSET + 3
CARD_FEATURE_DIM = CARD_SPECIAL_OFFSET + 4


def _enum_index(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_card_feature_table(
    cards: Iterable[Any],
    card_count: int,
) -> torch.Tensor:
    """Build the reference notebook's ``(card_count, 54)`` feature table."""
    if card_count < 1:
        raise ValueError("card_count must be positive")
    features = torch.zeros(
        (card_count, CARD_FEATURE_DIM),
        dtype=torch.float32,
    )
    populated = False
    for card in cards:
        card_id = int(card.cardId)
        if not 0 <= card_id < card_count:
            continue
        populated = True

        card_type = _enum_index(card.cardType)
        if card_type is not None and 0 <= card_type < CARD_TYPE_DIM:
            features[card_id, CARD_TYPE_OFFSET + card_type] = 1

        energy_type = _enum_index(card.energyType)
        if energy_type is not None and 0 <= energy_type < ENERGY_TYPE_DIM:
            features[
                card_id,
                CARD_ENERGY_TYPE_OFFSET + energy_type,
            ] = 1

        features[card_id, CARD_HP_INDEX] = float(card.hp) / 400
        features[card_id, CARD_RETREAT_INDEX] = (
            float(card.retreatCost) / 5
        )

        weakness = _enum_index(card.weakness)
        weakness_index = (
            weakness
            if weakness is not None and 0 <= weakness < ENERGY_TYPE_DIM
            else ENERGY_TYPE_DIM
        )
        features[
            card_id,
            CARD_WEAKNESS_OFFSET + weakness_index,
        ] = 1

        resistance = _enum_index(card.resistance)
        resistance_index = (
            resistance
            if resistance is not None and 0 <= resistance < ENERGY_TYPE_DIM
            else ENERGY_TYPE_DIM
        )
        features[
            card_id,
            CARD_RESISTANCE_OFFSET + resistance_index,
        ] = 1

        features[card_id, CARD_STAGE_OFFSET:CARD_SPECIAL_OFFSET] = (
            torch.tensor(
                [
                    float(card.basic),
                    float(card.stage1),
                    float(card.stage2),
                ]
            )
        )
        features[card_id, CARD_SPECIAL_OFFSET:CARD_FEATURE_DIM] = (
            torch.tensor(
                [
                    float(card.ex),
                    float(card.megaEx),
                    float(card.tera),
                    float(card.aceSpec),
                ]
            )
        )

    if not populated:
        raise RuntimeError("cg.api returned no valid card data")
    return features
