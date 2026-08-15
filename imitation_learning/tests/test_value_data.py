from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.feature_cache import (
    ATTACK_DYNAMIC_DIM,
    ENCODER_WORDS,
    GLOBAL_SUMMARY_DIM,
    HISTORY_STRUCTURAL_DIM,
    OPTION_NUMERIC_DIM,
    OPPONENT_SUMMARY_DIM,
    OWN_SUMMARY_DIM,
    PLAYER_RESULT_DRAW,
    PLAYER_RESULT_LOSS,
    PLAYER_RESULT_WIN,
    POKEMON_DYNAMIC_DIM,
    POKEMON_ENCODER_TOKENS,
    FeatureRecord,
    IndexBatch,
    MmapFeatureDataset,
    PackedShardWriter,
    _mix_episode_keys,
    stable_deck_key,
    stable_episode_key,
)
from value.data import build_value_splits, player_result_targets


SIGNATURE = {
    "card_count": 1267,
    "attack_count": 512,
    "encoder_size": 22000,
    "decoder_layout": "option-components-original16-plus-one-hot-v5",
    "max_actions": 64,
}


def _record(
    marker: int,
    episode: str,
    result: int,
    *,
    deck_marker: int = 1,
) -> FeatureRecord:
    offsets = [0] * ENCODER_WORDS
    offsets[-1] = 1
    return FeatureRecord(
        encoder_index=[marker, marker + 1],
        encoder_value=[1.0, 0.5],
        encoder_offset=offsets,
        encoder_pokemon_appear=[0] * POKEMON_ENCODER_TOKENS,
        own_summary=[float(marker)] * OWN_SUMMARY_DIM,
        opponent_summary=[float(marker)] * OPPONENT_SUMMARY_DIM,
        global_summary=[float(marker)] * GLOBAL_SUMMARY_DIM,
        option_categorical=[0] * 11,
        option_numeric=[0.0] * OPTION_NUMERIC_DIM,
        pokemon_dynamic=[0.0] * POKEMON_DYNAMIC_DIM,
        attack_dynamic=[0.0] * ATTACK_DYNAMIC_DIM,
        action_option_index=[0],
        action_option_offset=[0, 1],
        target=0,
        action_count=1,
        episode_key=stable_episode_key(episode),
        deck_key=stable_deck_key([deck_marker] * 60),
        player_result=result,
        history_select_type=[0, 0, 0],
        history_select_context=[0, 0, 0],
        history_valid=[0, 0, 0],
        history_option_categorical=[],
        history_structural=[],
        history_pokemon_dynamic=[],
        history_attack_dynamic=[],
        history_option_offset=[0, 0, 0, 0],
    )


def _write(path: Path, source: str, rows: list[FeatureRecord]) -> None:
    writer = PackedShardWriter(path, SIGNATURE, {"name": source})
    for row in rows:
        writer.add(row)
    writer.finalize()


def _dataset(tmp_path: Path) -> MmapFeatureDataset:
    _write(
        tmp_path / "7.1.cache",
        "7.1.jsonl.gz",
        [
            _record(1, "validation", PLAYER_RESULT_WIN, deck_marker=7),
            _record(2, "validation", PLAYER_RESULT_LOSS, deck_marker=7),
            _record(3, "validation", PLAYER_RESULT_DRAW, deck_marker=7),
            _record(4, "training", PLAYER_RESULT_WIN),
            _record(5, "training", PLAYER_RESULT_LOSS),
            _record(6, "isolation", PLAYER_RESULT_WIN),
            _record(7, "isolation", PLAYER_RESULT_LOSS),
        ],
    )
    _write(
        tmp_path / "7.2.cache",
        "7.2.jsonl.gz",
        [
            _record(8, "latest", PLAYER_RESULT_WIN, deck_marker=7),
            _record(9, "latest", PLAYER_RESULT_LOSS, deck_marker=7),
            _record(10, "latest-draw", PLAYER_RESULT_DRAW),
        ],
    )
    return MmapFeatureDataset(tmp_path, SIGNATURE)


def test_player_result_targets_map_win_draw_loss() -> None:
    actual = player_result_targets(
        np.array(
            [PLAYER_RESULT_WIN, PLAYER_RESULT_DRAW, PLAYER_RESULT_LOSS],
            dtype=np.uint8,
        )
    )
    np.testing.assert_array_equal(actual, [1.0, 0.0, -1.0])


def test_encoder_collation_omits_current_options_and_keeps_results(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    try:
        batch = dataset.collate_encoder(IndexBatch(np.array([0, 1, 2])))
        assert not hasattr(batch, "option_categorical")
        assert not hasattr(batch, "action_option_index")
        assert batch.own_summary.shape == (3, OWN_SUMMARY_DIM)
        np.testing.assert_array_equal(
            batch.player_result,
            [PLAYER_RESULT_WIN, PLAYER_RESULT_LOSS, PLAYER_RESULT_DRAW],
        )
    finally:
        dataset.close()


def test_value_splits_are_replay_level_and_retain_all_outcomes(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    validation_key = stable_episode_key("validation")
    training_key = stable_episode_key("training")
    threshold = int(0.5 * (1 << 32))
    seed = next(
        value
        for value in range(1000)
        if _mix_episode_keys(np.array([validation_key]), value)[0] < threshold
        and _mix_episode_keys(np.array([training_key]), value)[0] >= threshold
    )
    try:
        splits = build_value_splits(
            dataset,
            validation_ratio=0.5,
            validation_seed=seed,
            expert_episode_keys={
                (7, 1): {validation_key},
                (7, 2): {stable_episode_key("latest")},
            },
            top_deck_keys=(stable_deck_key([7] * 60),),
            isolation_episode_keys={
                "val_deck_isolation": {
                    (7, 1): {stable_episode_key("isolation")},
                    (7, 2): set(),
                }
            },
        )

        np.testing.assert_array_equal(splits.in_distribution, [0, 1, 2])
        np.testing.assert_array_equal(splits.train, [3, 4])
        np.testing.assert_array_equal(splits.isolation, [5, 6])
        np.testing.assert_array_equal(splits.latest, [7, 8, 9])
        np.testing.assert_array_equal(
            splits.in_distribution_masks["val_in_distribution_expert"],
            [True, True, True],
        )
        np.testing.assert_array_equal(
            splits.in_distribution_masks["val_in_distribution_deck1"],
            [True, True, True],
        )
        np.testing.assert_array_equal(
            splits.latest_masks["val_latest_expert_deck1"],
            [True, True, False],
        )
        assert splits.latest_date == (7, 2)
    finally:
        dataset.close()
