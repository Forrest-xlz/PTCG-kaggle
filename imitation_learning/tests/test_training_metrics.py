from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train import (
    ExponentialMovingAverage,
    policy_metrics,
    load_settings,
)


def test_ema_initializes_from_first_value_and_persists() -> None:
    ema = ExponentialMovingAverage(alpha=0.99)
    assert ema.update(2.0) == pytest.approx(2.0)
    assert ema.update(1.0) == pytest.approx(1.99)
    assert ema.update(0.0) == pytest.approx(1.9701)


def test_training_config_force_includes_manual_replay_date(
    tmp_path: Path,
) -> None:
    config_path = PROJECT_ROOT / "cfg" / "train.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "loser_augmentation" not in config["train"]
    assert config["train"]["include_all_dates"] == ["8.16"]
    copied_path = tmp_path / "train.yaml"
    copied_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    assert load_settings(copied_path).train.include_all_dates == ("8.16",)


def test_training_config_rejects_invalid_included_date(tmp_path: Path) -> None:
    config_path = PROJECT_ROOT / "cfg" / "train.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["train"]["include_all_dates"] = ["not-a-date"]
    invalid_path = tmp_path / "train.yaml"
    invalid_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="cannot parse month.day"):
        load_settings(invalid_path)


def test_policy_metrics_mask_invalid_actions_and_compute_topk() -> None:
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 100.0, 100.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    targets = torch.tensor([2, 3])
    action_counts = torch.tensor([3, 5])

    loss, metrics = policy_metrics(logits, targets, action_counts)

    assert torch.isfinite(loss)
    assert metrics.samples == 2
    assert metrics.top1_correct == 0
    assert metrics.top3_correct == 1
    assert metrics.top5_correct == 2
