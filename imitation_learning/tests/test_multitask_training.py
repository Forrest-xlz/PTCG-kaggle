import threading

import torch

from training.train import auxiliary_classification_metrics, prefetch_iterable


def test_masked_auxiliary_ce_ignores_invalid_rows() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    mask = torch.tensor([True, True, False])

    loss, correct, samples = auxiliary_classification_metrics(
        logits, targets, mask
    )

    assert loss.item() < 0.1
    assert isinstance(correct, torch.Tensor)
    assert isinstance(samples, torch.Tensor)
    assert correct.dtype == torch.int64
    assert samples.dtype == torch.int64
    assert correct.item() == 2
    assert samples.item() == 2


def test_empty_auxiliary_mask_returns_differentiable_zero() -> None:
    logits = torch.randn(2, 3, requires_grad=True)
    loss, correct, samples = auxiliary_classification_metrics(
        logits, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.bool)
    )
    loss.backward()
    assert loss.item() == 0.0
    assert correct.item() == 0
    assert samples.item() == 0
    assert logits.grad is not None


def test_masked_auxiliary_ce_does_not_backpropagate_invalid_rows() -> None:
    logits = torch.randn(3, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2])
    valid = torch.tensor([True, False, True])

    loss, _, _ = auxiliary_classification_metrics(logits, targets, valid)
    loss.backward()

    assert torch.count_nonzero(logits.grad[1]).item() == 0


def test_prefetch_iterable_requests_next_item_in_background() -> None:
    requested_second = threading.Event()
    release_second = threading.Event()

    def source():
        yield 1
        requested_second.set()
        release_second.wait(timeout=2)
        yield 2

    values = prefetch_iterable(source(), buffer_size=1)
    assert next(values) == 1
    assert requested_second.wait(timeout=1)
    release_second.set()
    assert next(values) == 2
