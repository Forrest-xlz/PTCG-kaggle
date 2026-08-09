from __future__ import annotations

from contextlib import nullcontext
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.expert_validation import ExpertLoserDateInfo
from training.feature_cache import CachedBatch, LoserAugmentationCounts
from training.feature_cache import (
    ENCODER_WORDS,
    GLOBAL_SUMMARY_DIM,
    ATTACK_DYNAMIC_DIM,
    HISTORY_STEPS,
    HISTORY_STRUCTURAL_DIM,
    OPTION_CATEGORICAL_DIM,
    OPPONENT_SUMMARY_DIM,
    OPTION_NUMERIC_DIM,
    OWN_SUMMARY_DIM,
    POKEMON_DYNAMIC_DIM,
)
from training.train import (
    ExponentialMovingAverage,
    PolicyMetrics,
    _log_validation,
    evaluate_dataset,
    policy_metrics,
    top_deck_subgroup_masks,
    format_loser_augmentation_line,
    select_loser_augmentation_dates,
)


def test_ema_initializes_from_first_value_and_persists() -> None:
    ema = ExponentialMovingAverage(alpha=0.99)
    assert ema.update(2.0) == pytest.approx(2.0)
    assert ema.update(1.0) == pytest.approx(1.99)
    assert ema.update(0.0) == pytest.approx(1.9701)


def test_loser_augmentation_dates_exclude_latest_validation_date() -> None:
    assert select_loser_augmentation_dates(
        [(7, 20), (7, 22), (7, 24), (7, 22)], recent_dates=2
    ) == ((7, 20), (7, 22))

    with pytest.raises(ValueError, match="recent_dates=3"):
        select_loser_augmentation_dates(
            [(7, 20), (7, 22), (7, 24)], recent_dates=3
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
        "after_validation_episodes=2 selected_train_episodes=1 "
        "loser_samples=17"
    )


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


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, encoder_index, encoder_value, encoder_offset, *args):
        self.forward_calls += 1
        batch_size = encoder_offset.numel() // ENCODER_WORDS
        return torch.arange(5, dtype=torch.float32).repeat(batch_size, 1)


class DummyDataset:
    def collate(self, index_batch):
        size = len(index_batch.global_ids)
        return CachedBatch(
            encoder_index=np.zeros(size, dtype=np.int32),
            encoder_value=np.ones(size, dtype=np.float16),
            encoder_offset=np.zeros(size * ENCODER_WORDS, dtype=np.int32),
            encoder_pokemon_appear=np.zeros(
                (size, 18), dtype=np.uint8
            ),
            own_summary=np.zeros((size, OWN_SUMMARY_DIM), dtype=np.float16),
            opponent_summary=np.zeros(
                (size, OPPONENT_SUMMARY_DIM), dtype=np.float16
            ),
            global_summary=np.zeros(
                (size, GLOBAL_SUMMARY_DIM), dtype=np.float16
            ),
            option_categorical=np.zeros(
                (size, OPTION_CATEGORICAL_DIM), dtype=np.int64
            ),
            option_numeric=np.zeros(
                (size, OPTION_NUMERIC_DIM), dtype=np.float16
            ),
            pokemon_dynamic=np.zeros(
                (size, POKEMON_DYNAMIC_DIM), dtype=np.float16
            ),
            attack_dynamic=np.zeros(
                (size, ATTACK_DYNAMIC_DIM), dtype=np.float16
            ),
            action_option_index=np.zeros(size, dtype=np.int64),
            action_option_offset=np.zeros(
                size * 64 + 1, dtype=np.int32
            ),
            target=np.full(size, 4, dtype=np.int64),
            action_count=np.full(size, 5, dtype=np.int64),
            history_select_type=np.zeros(
                (size, HISTORY_STEPS), dtype=np.uint8
            ),
            history_select_context=np.zeros(
                (size, HISTORY_STEPS), dtype=np.uint8
            ),
            history_valid=np.zeros(
                (size, HISTORY_STEPS), dtype=np.uint8
            ),
            history_option_categorical=np.empty(
                (0, OPTION_CATEGORICAL_DIM), dtype=np.int64
            ),
            history_structural=np.empty(
                (0, HISTORY_STRUCTURAL_DIM), dtype=np.int64
            ),
            history_pokemon_dynamic=np.empty(
                (0, POKEMON_DYNAMIC_DIM), dtype=np.float16
            ),
            history_attack_dynamic=np.empty(
                (0, ATTACK_DYNAMIC_DIM), dtype=np.float16
            ),
            history_option_offset=np.zeros(
                size * HISTORY_STEPS + 1, dtype=np.int32
            ),
        )


class DummyPrecision:
    def autocast(self):
        return nullcontext()


def test_validation_subgroups_share_model_forwards() -> None:
    model = CountingModel()
    result = evaluate_dataset(
        model=model,
        dataset=DummyDataset(),
        indices=np.arange(5, dtype=np.uint32),
        batch_size=2,
        device=torch.device("cpu"),
        precision=DummyPrecision(),
        subgroup_masks={
            "expert": np.asarray([True, False, True, False, True]),
            "top_deck": np.asarray([False, True, True, False, False]),
        },
    )

    assert model.forward_calls == 3
    assert result.overall.samples == 5
    assert result.subgroups["expert"].samples == 3
    assert result.subgroups["top_deck"].samples == 2


def test_top_deck_subgroup_names_follow_yaml_order() -> None:
    deck_masks = (
        np.asarray([True, False]),
        np.asarray([False, True]),
    )
    expert_masks = (
        np.asarray([False, False]),
        np.asarray([False, True]),
    )

    masks = top_deck_subgroup_masks(
        "val_in_distribution", deck_masks, expert_masks
    )

    assert list(masks) == [
        "val_in_distribution_deck1",
        "val_in_distribution_expert_deck1",
        "val_in_distribution_deck2",
        "val_in_distribution_expert_deck2",
    ]
    assert masks["val_in_distribution_deck1"] is deck_masks[0]
    assert masks["val_in_distribution_expert_deck2"] is expert_masks[1]


def test_validation_wandb_group_contains_only_policy_metrics() -> None:
    class Run:
        def __init__(self) -> None:
            self.payload = None

        def log(self, payload) -> None:
            self.payload = payload

    run = Run()
    _log_validation(
        "val_latest_deck1",
        PolicyMetrics(
            loss=1.25,
            top1_correct=2,
            top3_correct=3,
            top5_correct=4,
            samples=4,
        ),
        seconds=0.5,
        global_step=7,
        wandb_run=run,
    )

    assert set(run.payload) == {
        "val_latest_deck1/loss",
        "val_latest_deck1/top1_accuracy",
        "val_latest_deck1/top3_accuracy",
        "val_latest_deck1/top5_accuracy",
        "optimizer_step",
    }
