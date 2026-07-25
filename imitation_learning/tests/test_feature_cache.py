from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.feature_cache import (
    ENCODER_WORDS,
    FeatureRecord,
    MmapFeatureDataset,
    PackedShard,
    PackedShardWriter,
)


SIGNATURE = {
    "card_count": 1267,
    "attack_count": 512,
    "encoder_size": 22000,
    "decoder_size": 72745,
    "max_actions": 64,
}


def record(marker: int, action_count: int = 2) -> FeatureRecord:
    encoder_offset = [0] * ENCODER_WORDS
    encoder_offset[-1] = 1
    return FeatureRecord(
        encoder_index=[marker, marker + 1],
        encoder_value=[1.0, 0.25],
        encoder_offset=encoder_offset,
        decoder_index=[marker + 100, marker + 101],
        decoder_offset=[0, 1][:action_count],
        target=min(1, action_count - 1),
        action_count=action_count,
    )


def build_shard(path: Path, markers: list[int]) -> Path:
    writer = PackedShardWriter(path, SIGNATURE, {"name": path.stem})
    for marker in markers:
        writer.add(record(marker))
    writer.finalize()
    return path


def test_packed_shard_round_trip(tmp_path: Path) -> None:
    path = build_shard(tmp_path / "part-00000.cache", [7, 11])

    shard = PackedShard(path, expected_signature=SIGNATURE)
    try:
        assert len(shard) == 2
        sample = shard.sample(1)
        np.testing.assert_array_equal(sample.encoder_index, [11, 12])
        np.testing.assert_allclose(sample.encoder_value, [1.0, 0.25])
        assert sample.encoder_value.dtype == np.float16
        assert sample.encoder_offset.shape == (ENCODER_WORDS,)
        np.testing.assert_array_equal(sample.decoder_index, [111, 112])
        np.testing.assert_array_equal(sample.decoder_offset, [0, 1])
        assert sample.target == 1
        assert sample.action_count == 2
    finally:
        shard.close()


def test_writer_rejects_encoder_index_outside_uint16(tmp_path: Path) -> None:
    writer = PackedShardWriter(tmp_path / "bad.cache", SIGNATURE, {})
    bad = record(7)
    bad.encoder_index[0] = 65536
    with pytest.raises(ValueError, match="encoder_index"):
        writer.add(bad)
    writer.abort()


def test_dataset_global_shuffle_visits_every_sample_once(tmp_path: Path) -> None:
    build_shard(tmp_path / "a.cache", [1, 2, 3])
    build_shard(tmp_path / "b.cache", [4, 5])
    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    try:
        batches = list(
            dataset.iter_index_batches(
                batch_size=2, shuffle_mode="global", seed=123, max_samples=None
            )
        )
        global_ids = np.concatenate([batch.global_ids for batch in batches])
        np.testing.assert_array_equal(np.sort(global_ids), np.arange(5))
    finally:
        dataset.close()


def test_dataset_shard_shuffle_visits_every_sample_once(tmp_path: Path) -> None:
    build_shard(tmp_path / "a.cache", [1, 2, 3])
    build_shard(tmp_path / "b.cache", [4, 5])
    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    try:
        batches = list(
            dataset.iter_index_batches(
                batch_size=2, shuffle_mode="shard", seed=123, max_samples=None
            )
        )
        global_ids = np.concatenate([batch.global_ids for batch in batches])
        np.testing.assert_array_equal(np.sort(global_ids), np.arange(5))
    finally:
        dataset.close()


def test_collate_pads_decoder_offsets_without_decoder_values(tmp_path: Path) -> None:
    build_shard(tmp_path / "a.cache", [1, 2])
    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    try:
        index_batch = next(
            dataset.iter_index_batches(
                batch_size=2, shuffle_mode="global", seed=1, max_samples=None
            )
        )
        batch = dataset.collate(index_batch)
        assert batch.encoder_offset.shape == (2 * ENCODER_WORDS,)
        assert batch.decoder_offset.shape == (2 * 64,)
        assert batch.encoder_index.dtype == np.int32
        assert batch.decoder_index.dtype == np.int32
        assert batch.encoder_value.dtype == np.float16
        assert batch.target.dtype == np.int64
        assert batch.action_count.dtype == np.int64
    finally:
        dataset.close()
