from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train import build_lr_scheduler, should_trigger


def test_warmup_then_cosine_reaches_target_and_zero() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_lr_scheduler(
        optimizer, total_steps=6, warmup_steps=2
    )
    learning_rates = []
    for _ in range(6):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert learning_rates == pytest.approx(
        [0.5, 1.0, 1.0, 0.75, 0.25, 0.0],
        abs=1e-7,
    )


def test_step_trigger_uses_positive_optimizer_steps() -> None:
    assert should_trigger(100, 100)
    assert not should_trigger(100, 99)
    assert not should_trigger(100, 0)
