from dataclasses import replace

import numpy as np

from test_feature_cache import SIGNATURE, record
from training.feature_cache import (
    IndexBatch,
    MmapFeatureDataset,
    PackedShardWriter,
    load_opponent_deck_classes,
)


def test_opponent_deck_csv_requires_contiguous_classes(tmp_path) -> None:
    path = tmp_path / "classes.csv"
    path.write_text("class_id,deck_id\n0,D-A\n2,D-B\n", encoding="utf-8")

    try:
        load_opponent_deck_classes(path)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("non-contiguous class IDs must be rejected")


def test_auxiliary_labels_round_trip_through_mmap_cache(tmp_path) -> None:
    sample = replace(
        record(7),
        next_select_type=3,
        next_select_context=12,
        next_decision_valid=1,
        opponent_deck_class=4,
        opponent_deck_valid=1,
        final_own_prize_count=2,
    )
    shard_path = tmp_path / "sample.cache"
    writer = PackedShardWriter(shard_path, SIGNATURE, {"name": "7.1.jsonl.gz"})
    writer.add(sample)
    writer.finalize()

    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    batch = dataset.collate(IndexBatch(np.asarray([0], dtype=np.int64)))

    assert batch.next_select_type.tolist() == [3]
    assert batch.next_select_context.tolist() == [12]
    assert batch.next_decision_valid.tolist() == [1]
    assert batch.opponent_deck_class.tolist() == [4]
    assert batch.opponent_deck_valid.tolist() == [1]
    assert batch.final_own_prize_count.tolist() == [2]
