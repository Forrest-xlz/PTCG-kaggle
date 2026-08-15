"""Runtime policy masking shared by training and validation."""
from __future__ import annotations

import torch


def prepare_policy_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    action_counts: torch.Tensor,
    action_eligible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask invalid actions and drop rows whose recorded target is masked."""
    if logits.ndim != 2:
        raise ValueError("logits must be two-dimensional")
    batch_size, action_width = logits.shape
    if targets.shape != (batch_size,):
        raise ValueError("targets must align with logits")
    if action_counts.shape != (batch_size,):
        raise ValueError("action_counts must align with logits")
    if bool(((action_counts < 1) | (action_counts > action_width)).any()):
        raise ValueError("action_counts are outside the logits width")
    if bool(((targets < 0) | (targets >= action_counts)).any()):
        raise ValueError("targets are outside their action counts")

    positions = torch.arange(action_width, device=logits.device)
    allowed = positions.unsqueeze(0) < action_counts.unsqueeze(1)
    if action_eligible is None:
        retained = torch.ones(
            batch_size, dtype=torch.bool, device=logits.device
        )
    else:
        if action_eligible.shape != logits.shape:
            raise ValueError("action_eligible must match logits")
        allowed = allowed & action_eligible.to(dtype=torch.bool)
        retained = allowed.gather(1, targets.unsqueeze(1)).squeeze(1)

    allowed = allowed[retained]
    retained_logits = logits[retained]
    retained_targets = targets[retained]
    masked = retained_logits.masked_fill(
        ~allowed,
        torch.finfo(logits.dtype).min,
    )
    return masked, retained_targets, retained
