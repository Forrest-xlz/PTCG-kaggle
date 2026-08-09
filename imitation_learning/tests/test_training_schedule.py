from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.precision import PrecisionContext
from training.train import (
    ExponentialMovingAverage,
    build_lr_scheduler,
    load_epoch_checkpoint,
    should_trigger,
)


def _training_objects():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    precision = PrecisionContext("fp32", torch.device("cpu"))
    ema = {
        name: ExponentialMovingAverage(0.99)
        for name in ("loss", "top1", "top3", "top5")
    }
    return model, optimizer, scheduler, precision, ema


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


def test_epoch_checkpoint_restores_state_and_returns_next_epoch(tmp_path: Path) -> None:
    source = _training_objects()
    source_model, source_optimizer, source_scheduler, _, source_ema = source
    source_model(torch.ones(1, 2)).sum().backward()
    source_optimizer.step()
    source_scheduler.step()
    source_ema["loss"].update(1.25)

    checkpoint_path = tmp_path / "epoch-002.pt"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scheduler": source_scheduler.state_dict(),
            "scaler": {},
            "global_step": 20,
            "epoch": 2,
            "config": {"d_model": 8},
            "ema": {
                name: tracker.state_dict()
                for name, tracker in source_ema.items()
            },
            "history": [{"epoch": 1}, {"epoch": 2}],
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        },
        checkpoint_path,
    )

    target_model, target_optimizer, target_scheduler, precision, ema = (
        _training_objects()
    )
    state = load_epoch_checkpoint(
        path=checkpoint_path,
        model=target_model,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        precision=precision,
        ema=ema,
        expected_model_config={"d_model": 8},
        target_epochs=5,
    )

    assert state.start_epoch_index == 2
    assert state.global_step == 20
    assert state.history == [{"epoch": 1}, {"epoch": 2}]
    assert ema["loss"].value == pytest.approx(1.25)
    for source_parameter, target_parameter in zip(
        source_model.parameters(), target_model.parameters()
    ):
        torch.testing.assert_close(source_parameter, target_parameter)


def test_resume_rejects_step_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "step-00000020.pt"
    torch.save({}, checkpoint_path)
    model, optimizer, scheduler, precision, ema = _training_objects()

    with pytest.raises(ValueError, match="epoch-\\*\\.pt"):
        load_epoch_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            precision=precision,
            ema=ema,
            expected_model_config={"d_model": 8},
            target_epochs=5,
        )
