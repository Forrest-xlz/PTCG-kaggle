from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.action_mask import prepare_policy_batch


def test_mask_removes_bad_action_and_row_with_bad_target() -> None:
    masked, targets, retained = prepare_policy_batch(
        logits=torch.tensor([[9.0, 1.0], [9.0, 1.0]]),
        targets=torch.tensor([1, 0]),
        action_counts=torch.tensor([2, 2]),
        action_eligible=torch.tensor(
            [[False, True], [False, True]]
        ),
    )

    assert retained.tolist() == [True, False]
    assert targets.tolist() == [1]
    assert masked.argmax(dim=1).tolist() == [1]


def test_disabled_mask_preserves_original_rows_and_logits() -> None:
    logits = torch.tensor([[9.0, 1.0], [9.0, 1.0]])

    masked, targets, retained = prepare_policy_batch(
        logits=logits,
        targets=torch.tensor([1, 0]),
        action_counts=torch.tensor([2, 2]),
        action_eligible=None,
    )

    assert retained.tolist() == [True, True]
    assert targets.tolist() == [1, 0]
    torch.testing.assert_close(masked, logits)


def test_action_count_padding_is_always_masked() -> None:
    masked, targets, retained = prepare_policy_batch(
        logits=torch.tensor([[1.0, 2.0, 99.0]]),
        targets=torch.tensor([1]),
        action_counts=torch.tensor([2]),
        action_eligible=torch.tensor([[True, True, True]]),
    )

    assert retained.tolist() == [True]
    assert targets.tolist() == [1]
    assert masked.argmax(dim=1).tolist() == [1]


def test_all_masked_labels_return_an_empty_policy_batch() -> None:
    logits = torch.tensor([[3.0, 2.0]], requires_grad=True)

    masked, targets, retained = prepare_policy_batch(
        logits=logits,
        targets=torch.tensor([0]),
        action_counts=torch.tensor([2]),
        action_eligible=torch.tensor([[False, True]]),
    )

    assert retained.tolist() == [False]
    assert masked.shape == (0, 2)
    assert targets.shape == (0,)
