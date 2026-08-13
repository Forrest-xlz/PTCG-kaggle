import torch

from training.train import auxiliary_classification_metrics


def test_masked_auxiliary_ce_ignores_invalid_rows() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    mask = torch.tensor([True, True, False])

    loss, correct, samples = auxiliary_classification_metrics(
        logits, targets, mask
    )

    assert loss.item() < 0.1
    assert correct == 2
    assert samples == 2


def test_empty_auxiliary_mask_returns_differentiable_zero() -> None:
    logits = torch.randn(2, 3, requires_grad=True)
    loss, correct, samples = auxiliary_classification_metrics(
        logits, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.bool)
    )
    loss.backward()
    assert loss.item() == 0.0
    assert correct == 0
    assert samples == 0
    assert logits.grad is not None
