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
    parse_source_date,
    stable_episode_key,
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
        episode_key=stable_episode_key(f"episode-{marker}"),
    )


def build_shard(
    path: Path,
    markers: list[int],
    source_name: str = "7.1.jsonl.gz",
) -> Path:
    writer = PackedShardWriter(path, SIGNATURE, {"name": source_name})
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
        assert sample.episode_key == stable_episode_key("episode-11")
        assert shard.arrays["episode_key"].dtype == np.dtype("<u4")
    finally:
        shard.close()


def test_writer_rejects_encoder_index_outside_uint16(tmp_path: Path) -> None:
    writer = PackedShardWriter(tmp_path / "bad.cache", SIGNATURE, {})
    bad = record(7)
    bad.encoder_index[0] = 65536
    with pytest.raises(ValueError, match="encoder_index"):
        writer.add(bad)
    writer.abort()


def test_episode_key_is_stable_for_equivalent_ids() -> None:
    assert stable_episode_key(12345) == stable_episode_key("12345")
    assert 0 <= stable_episode_key("episode-a") <= np.iinfo(np.uint32).max


def test_source_dates_sort_numerically() -> None:
    assert parse_source_date("7.19.jsonl.gz") > parse_source_date("7.5.jsonl.gz")
    assert parse_source_date("7.1.part-00000.cache") == (7, 1)


def test_dataset_global_shuffle_visits_every_sample_once(tmp_path: Path) -> None:
    build_shard(tmp_path / "a.cache", [1, 2, 3])
    build_shard(tmp_path / "b.cache", [4, 5])
    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    try:
        batches = list(
            dataset.iter_index_batches(
                indices=np.arange(5, dtype=np.uint32),
                batch_size=2,
                seed=123,
                shuffle=True,
            )
        )
        global_ids = np.concatenate([batch.global_ids for batch in batches])
        np.testing.assert_array_equal(np.sort(global_ids), np.arange(5))
    finally:
        dataset.close()


def test_splits_hold_out_latest_and_keep_replays_together(
    tmp_path: Path,
) -> None:
    build_shard(
        tmp_path / "old-a.cache",
        list(range(1, 51)) * 2,
        source_name="7.5.jsonl.gz",
    )
    build_shard(
        tmp_path / "old-b.cache",
        list(range(51, 101)) * 2,
        source_name="7.19.jsonl.gz",
    )
    build_shard(
        tmp_path / "latest.cache",
        [101, 102, 103],
        source_name="7.24.jsonl.gz",
    )
    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    try:
        splits = dataset.build_splits(
            validation_ratio=0.5, validation_seed=123
        )
        assert splits.latest_date == (7, 24)
        expected_latest = np.concatenate(
            [
                np.arange(dataset.starts[i], dataset.ends[i])
                for i, date in enumerate(dataset.shard_dates)
                if date == (7, 24)
            ]
        )
        np.testing.assert_array_equal(splits.latest, expected_latest)
        assert not set(splits.latest) & set(splits.train)
        assert not set(splits.latest) & set(splits.in_distribution)

        memberships = {}
        for name, ids in (
            ("train", splits.train),
            ("validation", splits.in_distribution),
        ):
            for global_id in ids:
                shard_id = np.searchsorted(
                    dataset.ends, global_id, side="right"
                )
                local_id = int(global_id - dataset.starts[shard_id])
                key = dataset.shards[shard_id].sample(local_id).episode_key
                memberships.setdefault(key, set()).add(name)
        assert all(len(groups) == 1 for groups in memberships.values())
    finally:
        dataset.close()


def test_collate_pads_decoder_offsets_without_decoder_values(tmp_path: Path) -> None:
    build_shard(tmp_path / "a.cache", [1, 2])
    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    try:
        index_batch = next(
            dataset.iter_index_batches(
                indices=np.arange(2, dtype=np.uint32),
                batch_size=2,
                seed=1,
                shuffle=True,
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
