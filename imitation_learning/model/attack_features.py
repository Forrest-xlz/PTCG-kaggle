"""Static attack features shared by training and inference."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from model.card_features import ENERGY_TYPE_DIM


ATTACK_DAMAGE_INDEX = 0
ATTACK_ENERGY_OFFSET = 1
ATTACK_ENERGY_COUNT_INDEX = ATTACK_ENERGY_OFFSET + ENERGY_TYPE_DIM
ATTACK_FEATURE_DIM = ATTACK_ENERGY_COUNT_INDEX + 1


def build_attack_feature_table(
    attacks: Iterable[Any],
    attack_count: int,
) -> torch.Tensor:
    """Build the reference notebook's static attack feature table."""
    if attack_count < 1:
        raise ValueError("attack_count must be positive")
    features = torch.zeros(
        (attack_count, ATTACK_FEATURE_DIM),
        dtype=torch.float32,
    )
    populated = False
    for attack in attacks:
        attack_id = int(attack.attackId)
        if not 0 <= attack_id < attack_count:
            continue
        populated = True
        features[attack_id, ATTACK_DAMAGE_INDEX] = float(attack.damage) / 300
        energies = list(attack.energies or [])
        for energy in energies:
            energy_type = int(energy)
            if 0 <= energy_type < ENERGY_TYPE_DIM:
                features[
                    attack_id,
                    ATTACK_ENERGY_OFFSET + energy_type,
                ] += 1
        features[attack_id, ATTACK_ENERGY_COUNT_INDEX] = len(energies) / 5
    if not populated:
        raise RuntimeError("cg.api returned no valid attack data")
    return features
