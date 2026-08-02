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

from training.feature_cache import CachedBatch
from training.feature_cache import (
    ENCODER_WORDS,
    GLOBAL_SUMMARY_DIM,
    OPPONENT_SUMMARY_DIM,
    OPTION_CATEGORICAL_DIM,
    OPTION_NUMERIC_DIM,
    OWN_SUMMARY_DIM,
)
from training.train import (
    ExponentialMovingAverage,
    evaluate_dataset,
    policy_metrics,
)


def test_ema_initializes_from_first_value_and_persists() -> None:
    ema = ExponentialMovingAverage(alpha=0.99)
    assert ema.update(2.0) == pytest.approx(2.0)
    assert ema.update(1.0) == pytest.approx(1.99)
    assert ema.update(0.0) == pytest.approx(1.9701)


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

    def forward(
        self,
        encoder_index,
        encoder_value,
        encoder_offset,
        own_summary,
        opponent_summary,
        global_summary,
        option_categorical,
        option_numeric,
        action_option_index,
        action_option_offset,
    ):
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
            action_option_index=np.zeros(size, dtype=np.int64),
            action_option_offset=np.zeros(
                size * 64 + 1, dtype=np.int32
            ),
            target=np.full(size, 4, dtype=np.int64),
            action_count=np.full(size, 5, dtype=np.int64),
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
