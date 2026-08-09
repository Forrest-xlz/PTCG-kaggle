from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.feature_cache import (  # noqa: E402
    ATTACK_DYNAMIC_DIM,
    ENCODER_WORDS,
    GLOBAL_SUMMARY_DIM,
    HISTORY_STEPS,
    HISTORY_STRUCTURAL_DIM,
    OPTION_CATEGORICAL_DIM,
    OPTION_NUMERIC_DIM,
    OPPONENT_SUMMARY_DIM,
    OWN_SUMMARY_DIM,
    POKEMON_DYNAMIC_DIM,
    POKEMON_ENCODER_TOKENS,
    FeatureRecord,
    IndexBatch,
    MmapFeatureDataset,
    PackedShard,
    PackedShardWriter,
)


def _record() -> FeatureRecord:
    return FeatureRecord(
        encoder_index=[1],
        encoder_value=[1.0],
        encoder_offset=[0] * ENCODER_WORDS,
        encoder_pokemon_appear=[0] * POKEMON_ENCODER_TOKENS,
        own_summary=[0.0] * OWN_SUMMARY_DIM,
        opponent_summary=[0.0] * OPPONENT_SUMMARY_DIM,
        global_summary=[0.0] * GLOBAL_SUMMARY_DIM,
        option_categorical=[0] * OPTION_CATEGORICAL_DIM,
        option_numeric=[0.0] * OPTION_NUMERIC_DIM,
        pokemon_dynamic=[0.0] * POKEMON_DYNAMIC_DIM,
        attack_dynamic=[0.0] * ATTACK_DYNAMIC_DIM,
        action_option_index=[0],
        action_option_offset=[0, 1],
        target=0,
        action_count=1,
        episode_key=7,
        deck_key=11,
        history_select_type=[0, 1, 9],
        history_select_context=[0, 21, 43],
        history_valid=[0, 1, 1],
        history_option_categorical=[1] * (3 * OPTION_CATEGORICAL_DIM),
        history_structural=[2] * (3 * HISTORY_STRUCTURAL_DIM),
        history_pokemon_dynamic=[3.0] * (3 * POKEMON_DYNAMIC_DIM),
        history_attack_dynamic=[4.0] * (3 * ATTACK_DYNAMIC_DIM),
        history_option_offset=[0, 0, 2, 3],
    )


def test_history_round_trip_and_collation(tmp_path: Path) -> None:
    destination = tmp_path / "history.part-00000.cache"
    writer = PackedShardWriter(
        destination,
        {"history": "v1"},
        {"name": "7.1.jsonl.gz"},
    )
    writer.add(_record())
    writer.finalize()

    shard = PackedShard(destination, expected_signature={"history": "v1"})
    sample = shard.sample(0)
    np.testing.assert_array_equal(sample.history_valid, [0, 1, 1])
    np.testing.assert_array_equal(sample.history_option_offset, [0, 0, 2, 3])
    assert sample.history_option_categorical.shape == (
        3,
        OPTION_CATEGORICAL_DIM,
    )
    assert sample.history_structural.shape == (3, HISTORY_STRUCTURAL_DIM)
    assert sample.history_pokemon_dynamic.shape == (3, POKEMON_DYNAMIC_DIM)
    assert sample.history_attack_dynamic.shape == (3, ATTACK_DYNAMIC_DIM)
    shard.close()

    dataset = MmapFeatureDataset(tmp_path, expected_signature={"history": "v1"})
    batch = dataset.collate(IndexBatch(np.asarray([0], dtype=np.uint32)))
    assert batch.history_valid.shape == (1, HISTORY_STEPS)
    np.testing.assert_array_equal(batch.history_option_offset, [0, 0, 2, 3])
    dataset.close()
