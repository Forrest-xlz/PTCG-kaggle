from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.expert_validation import ExpertLoserDateInfo
from training.feature_cache import LoserAugmentationCounts
from training.train import (
    ExponentialMovingAverage,
    format_loser_augmentation_line,
    load_settings,
    policy_metrics,
    select_loser_augmentation_dates,
)


def test_ema_initializes_from_first_value_and_persists() -> None:
    ema = ExponentialMovingAverage(alpha=0.99)
    assert ema.update(2.0) == pytest.approx(2.0)
    assert ema.update(1.0) == pytest.approx(1.99)
    assert ema.update(0.0) == pytest.approx(1.9701)


def test_loser_augmentation_dates_include_latest_cached_date() -> None:
    assert select_loser_augmentation_dates(
        [(7, 20), (7, 22), (7, 24), (7, 22)], recent_dates=2
    ) == ((7, 22), (7, 24))

    with pytest.raises(ValueError, match="recent_dates=4"):
        select_loser_augmentation_dates(
            [(7, 20), (7, 22), (7, 24)], recent_dates=4
        )


def test_loser_augmentation_line_reports_each_filter_stage() -> None:
    line = format_loser_augmentation_line(
        ExpertLoserDateInfo(
            date=(7, 21),
            cutoff=1184.0,
            participant_count=18_420,
            episode_count=9_210,
            eligible_episode_keys=frozenset({1, 2, 3}),
        ),
        LoserAugmentationCounts(
            score_eligible_episodes=3,
            after_validation_episodes=2,
            selected_train_episodes=1,
            loser_samples=17,
        ),
    )

    assert line == (
        "loser_aug_date=7.21 cutoff=1184 participant_scores=18,420 "
        "episodes=9,210 score_eligible_episodes=3 "
        "selected_train_episodes=1 loser_samples=17"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("enabled", 1, "enabled"),
        ("recent_dates", 0, "recent_dates"),
        ("expert_ratio", 0.0, "expert_ratio"),
        ("expert_ratio", 1.1, "expert_ratio"),
    ],
)
def test_loser_augmentation_settings_validate_ranges(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    config_path = PROJECT_ROOT / "cfg" / "train.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["train"]["loser_augmentation"][field] = value
    invalid_path = tmp_path / "train.yaml"
    invalid_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
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
