from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beam_search.model_agent import (
    PolicyHistory,
    history_tensors,
    normalize_policy,
)
from model.features import (
    ATTACK_DYNAMIC_DIM,
    HISTORY_STEPS,
    HISTORY_STRUCTURAL_DIM,
    OPTION_CATEGORICAL_DIM,
    POKEMON_DYNAMIC_DIM,
    HistoryActionFeatures,
)


def _history_action(value: int, rows: int = 1) -> HistoryActionFeatures:
    return HistoryActionFeatures(
        select_type=value,
        select_context=value + 1,
        option_categorical=np.full(
            (rows, OPTION_CATEGORICAL_DIM), value, dtype=np.int64
        ),
        structural=np.full(
            (rows, HISTORY_STRUCTURAL_DIM), value, dtype=np.int64
        ),
        pokemon_dynamic=np.full(
            (rows, POKEMON_DYNAMIC_DIM), value, dtype=np.float32
        ),
        attack_dynamic=np.full(
            (rows, ATTACK_DYNAMIC_DIM), value, dtype=np.float32
        ),
    )


def test_policy_history_clone_is_independent_and_bounded() -> None:
    original = PolicyHistory()
    for value in range(HISTORY_STEPS + 1):
        original.append(_history_action(value))
    clone = original.clone()
    clone.append(_history_action(9))

    assert [action.select_type for action in original.actions] == [1, 2, 3]
    assert [action.select_type for action in clone.actions] == [2, 3, 9]


def test_history_tensors_right_align_actions_and_offsets() -> None:
    history = PolicyHistory()
    history.append(_history_action(4, rows=2))
    history.append(_history_action(7, rows=1))

    tensors = history_tensors(history, torch.device("cpu"))
    select_type, select_context, valid = tensors[:3]
    categorical, structural, pokemon, attack, offsets = tensors[3:]

    assert select_type.tolist() == [[0, 4, 7]]
    assert select_context.tolist() == [[0, 5, 8]]
    assert valid.tolist() == [[False, True, True]]
    assert categorical.shape == (3, OPTION_CATEGORICAL_DIM)
    assert structural.shape == (3, HISTORY_STRUCTURAL_DIM)
    assert pokemon.shape == (3, POKEMON_DYNAMIC_DIM)
    assert attack.shape == (3, ATTACK_DYNAMIC_DIM)
    assert offsets.tolist() == [0, 0, 2, 3]


def test_normalize_policy_uses_fp32_softmax_and_clamped_logs() -> None:
    actions = ([0], [1], [2])
    output = normalize_policy(actions, torch.tensor([[0.0, 1.0, 2.0]]))

    assert output.actions == actions
    assert output.probabilities.dtype == torch.float32
    assert output.probabilities.sum().item() == pytest.approx(1.0)
    assert int(output.probabilities.argmax()) == 2
    np.testing.assert_allclose(
        output.log_probabilities.cpu().numpy(),
        np.log(output.probabilities.cpu().numpy()),
        rtol=1e-6,
    )


def test_normalize_policy_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="action count"):
        normalize_policy(([0],), torch.tensor([[0.0, 1.0]]))
