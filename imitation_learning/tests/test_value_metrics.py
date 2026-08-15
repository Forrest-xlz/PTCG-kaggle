from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.feature_cache import (
    EncoderBatch,
    PLAYER_RESULT_DRAW,
    PLAYER_RESULT_LOSS,
    PLAYER_RESULT_WIN,
)
from value.metrics import RegressionAccumulator, evaluate_value_dataset


def test_regression_accumulator_computes_dataset_exact_metrics() -> None:
    accumulator = RegressionAccumulator()
    accumulator.update(np.array([0.5, -0.5]), np.array([1.0, -1.0]))
    accumulator.update(np.array([0.0]), np.array([0.0]))

    metrics = accumulator.finalize()

    assert metrics.samples == 3
    assert metrics.rmse == pytest.approx(math.sqrt(0.5 / 3))
    assert metrics.explained_variance == pytest.approx(0.75)
    assert metrics.target_mean == pytest.approx(0.0)
    assert metrics.prediction_mean == pytest.approx(0.0)


def test_explained_variance_is_nan_for_constant_targets() -> None:
    accumulator = RegressionAccumulator()
    accumulator.update(torch.tensor([0.0, 0.5]), torch.tensor([1.0, 1.0]))

    assert math.isnan(accumulator.finalize().explained_variance)


class Precision:
    def autocast(self):
        return nullcontext()


class Dataset:
    def collate_encoder(self, index_batch):
        ids = np.asarray(index_batch.global_ids, dtype=np.int64)
        batch = len(ids)
        predictions = np.array([0.5, -0.5, 0.1, -0.2])[ids]
        results = np.array(
            [
                PLAYER_RESULT_WIN,
                PLAYER_RESULT_LOSS,
                PLAYER_RESULT_DRAW,
                PLAYER_RESULT_LOSS,
            ],
            dtype=np.uint8,
        )[ids]
        return EncoderBatch(
            encoder_index=np.zeros(batch * 26, dtype=np.int32),
            encoder_value=np.ones(batch * 26, dtype=np.float16),
            encoder_offset=np.arange(batch * 26, dtype=np.int32),
            encoder_pokemon_appear=np.zeros((batch, 18), dtype=np.uint8),
            own_summary=np.pad(
                predictions[:, None], ((0, 0), (0, 68))
            ).astype(np.float16),
            opponent_summary=np.zeros((batch, 71), dtype=np.float16),
            global_summary=np.zeros((batch, 73), dtype=np.float16),
            history_select_type=np.zeros((batch, 3), dtype=np.uint8),
            history_select_context=np.zeros((batch, 3), dtype=np.uint8),
            history_valid=np.zeros((batch, 3), dtype=np.uint8),
            history_option_categorical=np.empty((0, 11), dtype=np.int64),
            history_structural=np.empty((0, 8), dtype=np.int64),
            history_pokemon_dynamic=np.empty((0, 46), dtype=np.float16),
            history_attack_dynamic=np.empty((0, 6), dtype=np.float16),
            history_option_offset=np.zeros(batch * 3 + 1, dtype=np.int32),
            player_result=results,
        )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *inputs):
        self.calls += 1
        own_summary = inputs[4]
        return own_summary[:, 0]


def test_evaluation_accumulates_subgroups_without_extra_forward_passes() -> None:
    model = Model()
    result = evaluate_value_dataset(
        model=model,
        dataset=Dataset(),
        indices=np.arange(4),
        batch_size=2,
        device=torch.device("cpu"),
        precision=Precision(),
        subgroup_masks={"expert": np.array([True, False, True, False])},
    )

    assert model.calls == 2
    assert result.overall.samples == 4
    assert result.subgroups["expert"].samples == 2
    expected = np.sqrt(np.mean((np.array([0.5, 0.1]) - [1.0, 0.0]) ** 2))
    assert result.subgroups["expert"].rmse == pytest.approx(expected, abs=1e-4)


def test_evaluation_rejects_misaligned_or_empty_subgroups() -> None:
    common = dict(
        model=Model(),
        dataset=Dataset(),
        indices=np.arange(4),
        batch_size=2,
        device=torch.device("cpu"),
        precision=Precision(),
    )
    with pytest.raises(ValueError, match="align"):
        evaluate_value_dataset(
            **common, subgroup_masks={"bad": np.ones(3, dtype=np.bool_)}
        )
    with pytest.raises(ValueError, match="empty"):
        evaluate_value_dataset(
            **common,
            subgroup_masks={"empty": np.zeros(4, dtype=np.bool_)},
        )
