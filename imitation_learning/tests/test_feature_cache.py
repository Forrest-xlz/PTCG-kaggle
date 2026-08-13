from __future__ import annotations

import csv
import io
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.feature_cache import (
    ENCODER_WORDS,
    POKEMON_ENCODER_TOKENS,
    GLOBAL_SUMMARY_DIM,
    OPPONENT_SUMMARY_DIM,
    ATTACK_DYNAMIC_DIM,
    HISTORY_STRUCTURAL_DIM,
    OPTION_CATEGORICAL_DIM,
    OPTION_NUMERIC_DIM,
    POKEMON_DYNAMIC_DIM,
    OWN_SUMMARY_DIM,
    PLAYER_RESULT_DRAW,
    PLAYER_RESULT_LOSS,
    PLAYER_RESULT_WIN,
    FeatureRecord,
    MmapFeatureDataset,
    PackedShard,
    PackedShardWriter,
    parse_source_date,
    stable_deck_key,
    stable_episode_key,
)
from training.expert_validation import load_expert_date_info


SIGNATURE = {
    "card_count": 1267,
    "attack_count": 512,
    "encoder_size": 22000,
    "decoder_layout": "option-components-original16-plus-one-hot-v5",
    "max_actions": 64,
}


def record(
    marker: int,
    action_count: int = 2,
    *,
    episode_id: str | None = None,
    player_result: int = PLAYER_RESULT_WIN,
) -> FeatureRecord:
    encoder_offset = [0] * ENCODER_WORDS
    encoder_offset[-1] = 1
    action_memberships = [[0, 1], [0]][:action_count]
    action_option_index = [
        index for action in action_memberships for index in action
    ]
    action_option_offset = [0]
    for action in action_memberships:
        action_option_offset.append(
            action_option_offset[-1] + len(action)
        )
    return FeatureRecord(
        encoder_index=[marker, marker + 1],
        encoder_value=[1.0, 0.25],
        encoder_offset=encoder_offset,
        encoder_pokemon_appear=(
            [2, 1] + [0] * (POKEMON_ENCODER_TOKENS - 2)
        ),
        own_summary=[float(marker)] * OWN_SUMMARY_DIM,
        opponent_summary=[float(marker + 1)] * OPPONENT_SUMMARY_DIM,
        global_summary=[float(marker + 2)] * GLOBAL_SUMMARY_DIM,
        option_categorical=(
            [8, 21, marker, marker + 1, 512, 0, 0, 1, 1, 1, 0]
            + [13, 21, 1267, 1267, 512, 0, 0, 0, 0, 0, 0]
        ),
        option_numeric=[float(marker) / 100] * (2 * OPTION_NUMERIC_DIM),
        pokemon_dynamic=[0.0] * (2 * POKEMON_DYNAMIC_DIM),
        attack_dynamic=[0.0] * (2 * ATTACK_DYNAMIC_DIM),
        history_select_type=[0, 0, 0],
        history_select_context=[0, 0, 0],
        history_valid=[0, 0, 0],
        history_option_categorical=[],
        history_structural=[],
        history_pokemon_dynamic=[],
        history_attack_dynamic=[],
        history_option_offset=[0, 0, 0, 0],
        action_option_index=action_option_index,
        action_option_offset=action_option_offset,
        target=min(1, action_count - 1),
        action_count=action_count,
        episode_key=stable_episode_key(episode_id or f"episode-{marker}"),
        deck_key=stable_deck_key([marker] * 60),
        player_result=player_result,
    )


def build_shard(
    path: Path,
    markers: list[int],
    source_name: str = "7.1.jsonl.gz",
    deck_markers: list[int] | None = None,
) -> Path:
    if deck_markers is not None and len(deck_markers) != len(markers):
        raise ValueError("deck_markers must align with markers")
    writer = PackedShardWriter(path, SIGNATURE, {"name": source_name})
    for index, marker in enumerate(markers):
        sample = record(marker)
        if deck_markers is not None:
            sample = replace(
                sample,
                deck_key=stable_deck_key([deck_markers[index]] * 60),
            )
        writer.add(sample)
    writer.finalize()
    return path


def write_manifest_archive(
    path: Path, rows: list[dict[str, object]]
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "episode_id",
            "min_score",
            "sum_score",
            "agent_count",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.csv", buffer.getvalue())


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
        np.testing.assert_array_equal(
            sample.encoder_pokemon_appear,
            [2, 1] + [0] * (POKEMON_ENCODER_TOKENS - 2),
        )
        assert sample.own_summary.shape == (OWN_SUMMARY_DIM,)
        assert sample.opponent_summary.shape == (OPPONENT_SUMMARY_DIM,)
        assert sample.global_summary.shape == (GLOBAL_SUMMARY_DIM,)
        assert sample.own_summary.dtype == np.float16
        np.testing.assert_array_equal(sample.own_summary, 11.0)
        np.testing.assert_array_equal(sample.opponent_summary, 12.0)
        np.testing.assert_array_equal(sample.global_summary, 13.0)
        assert sample.option_categorical.shape == (2, OPTION_CATEGORICAL_DIM)
        assert sample.option_numeric.shape == (2, OPTION_NUMERIC_DIM)
        assert sample.option_numeric.dtype == np.float16
        np.testing.assert_array_equal(
            sample.option_categorical[0, :5], [8, 21, 11, 12, 512]
        )
        np.testing.assert_array_equal(sample.action_option_index, [0, 1, 0])
        np.testing.assert_array_equal(sample.action_option_offset, [0, 2, 3])
        assert sample.target == 1
        assert sample.action_count == 2
        assert sample.episode_key == stable_episode_key("episode-11")
        assert sample.deck_key == stable_deck_key([11] * 60)
        assert shard.arrays["episode_key"].dtype == np.dtype("<u4")
        assert shard.arrays["deck_key"].dtype == np.dtype("<u8")
    finally:
        shard.close()


def test_player_result_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "results.cache"
    writer = PackedShardWriter(path, SIGNATURE, {"name": "7.1.jsonl.gz"})
    writer.add(record(1, player_result=PLAYER_RESULT_WIN))
    writer.add(record(2, player_result=PLAYER_RESULT_LOSS))
    writer.add(record(3, player_result=PLAYER_RESULT_DRAW))
    writer.finalize()

    shard = PackedShard(path, expected_signature=SIGNATURE)
    try:
        np.testing.assert_array_equal(
            shard.arrays["player_result"],
            [PLAYER_RESULT_WIN, PLAYER_RESULT_LOSS, PLAYER_RESULT_DRAW],
        )
        assert shard.sample(1).player_result == PLAYER_RESULT_LOSS
    finally:
        shard.close()


def test_splits_keep_validation_winner_only_and_add_qualified_losses(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "7.23-part.cache"
    writer = PackedShardWriter(
        old_path, SIGNATURE, {"name": "7.23.jsonl.gz"}
    )
    writer.add(
        record(1, episode_id="qualified", player_result=PLAYER_RESULT_WIN)
    )
    writer.add(
        record(2, episode_id="qualified", player_result=PLAYER_RESULT_LOSS)
    )
    writer.add(
        record(3, episode_id="unqualified", player_result=PLAYER_RESULT_WIN)
    )
    writer.add(
        record(4, episode_id="unqualified", player_result=PLAYER_RESULT_LOSS)
    )
    writer.add(record(5, episode_id="draw", player_result=PLAYER_RESULT_DRAW))
    for marker in range(10, 210):
        writer.add(record(marker))
    writer.finalize()

    latest_path = tmp_path / "7.24-part.cache"
    writer = PackedShardWriter(
        latest_path, SIGNATURE, {"name": "7.24.jsonl.gz"}
    )
    writer.add(record(300, episode_id="latest", player_result=PLAYER_RESULT_WIN))
    writer.add(
        record(301, episode_id="latest", player_result=PLAYER_RESULT_LOSS)
    )
    writer.add(record(302, episode_id="latest-draw", player_result=PLAYER_RESULT_DRAW))
    writer.finalize()

    dataset = MmapFeatureDataset(tmp_path, SIGNATURE)
    try:
        validation_seed = next(
            seed
            for seed in range(10_000)
            if stable_episode_key("qualified")
            not in {
                int(dataset.shards[0].arrays["episode_key"][index])
                for index in dataset.build_splits(
                    validation_ratio=0.05,
                    validation_seed=seed,
                ).in_distribution
            }
        )
        splits = dataset.build_splits(
            validation_ratio=0.05,
            validation_seed=validation_seed,
            loser_episode_keys={
                (7, 23): {stable_episode_key("qualified")}
            },
        )

        old_count = len(dataset.shards[0])
        train_old_ids = splits.train[splits.train < old_count]
        train_results = dataset.shards[0].arrays["player_result"][train_old_ids]
        assert PLAYER_RESULT_LOSS in train_results
        assert PLAYER_RESULT_DRAW not in train_results
        assert 3 not in train_old_ids
        assert 4 not in train_old_ids

        latest_local_ids = splits.latest - old_count
        np.testing.assert_array_equal(latest_local_ids, [0])
        assert dataset.shards[1].arrays["player_result"][latest_local_ids[0]] == (
            PLAYER_RESULT_WIN
        )

        counts = splits.loser_augmentation_counts[(7, 23)]
        assert counts.score_eligible_episodes == 1
        assert counts.after_validation_episodes == 1
        assert counts.selected_train_episodes == 1
        assert counts.loser_samples == 1
        assert splits.loser_augmentation_replays == 1
        assert splits.loser_augmentation_samples == 1
    finally:
        dataset.close()


def test_full_training_selection_uses_every_date_and_qualified_losses(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "7.23-part.cache"
    writer = PackedShardWriter(
        old_path, SIGNATURE, {"name": "7.23.jsonl.gz"}
    )
    writer.add(record(1, episode_id="old", player_result=PLAYER_RESULT_WIN))
    writer.add(record(2, episode_id="old", player_result=PLAYER_RESULT_LOSS))
    writer.finalize()

    latest_path = tmp_path / "7.24-part.cache"
    writer = PackedShardWriter(
        latest_path, SIGNATURE, {"name": "7.24.jsonl.gz"}
    )
    writer.add(
        record(3, episode_id="latest", player_result=PLAYER_RESULT_WIN)
    )
    writer.add(
        record(4, episode_id="latest", player_result=PLAYER_RESULT_LOSS)
    )
    writer.add(
        record(5, episode_id="draw", player_result=PLAYER_RESULT_DRAW)
    )
    writer.finalize()

    dataset = MmapFeatureDataset(tmp_path, SIGNATURE)
    try:
        selection = dataset.build_training_indices(
            loser_episode_keys={
                (7, 24): {stable_episode_key("latest")},
            },
        )

        np.testing.assert_array_equal(selection.train, [0, 2, 3])
        assert selection.eligible_train_samples == 3
        assert selection.eligible_train_replays == 2
        assert selection.selected_train_replays == 2
        assert selection.loser_augmentation_samples == 1
        assert selection.loser_augmentation_replays == 1
        counts = selection.loser_augmentation_counts[(7, 24)]
        assert counts.score_eligible_episodes == 1
        assert counts.selected_train_episodes == 1
        assert counts.loser_samples == 1
    finally:
        dataset.close()


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


def test_deck_key_ignores_order_and_preserves_multiplicity() -> None:
    deck = list(range(60))
    assert stable_deck_key(deck) == stable_deck_key(reversed(deck))
    changed = deck.copy()
    changed[-1] = changed[-2]
    assert stable_deck_key(deck) != stable_deck_key(changed)


def test_deck_key_requires_exactly_60_cards() -> None:
    with pytest.raises(ValueError, match="exactly 60"):
        stable_deck_key([1] * 59)


def test_source_dates_sort_numerically() -> None:
    assert parse_source_date("7.19.jsonl.gz") > parse_source_date("7.5.jsonl.gz")
    assert parse_source_date("7.1.part-00000.cache") == (7, 1)


def test_expert_cutoff_uses_both_scores_and_includes_ties(
    tmp_path: Path,
) -> None:
    write_manifest_archive(
        tmp_path / "7.24.zip",
        [
            {
                "episode_id": "a",
                "min_score": 10,
                "sum_score": 100,
                "agent_count": 2,
            },
            {
                "episode_id": "b",
                "min_score": 20,
                "sum_score": 100,
                "agent_count": 2,
            },
            {
                "episode_id": "c",
                "min_score": 30,
                "sum_score": 110,
                "agent_count": 2,
            },
        ],
    )

    info = load_expert_date_info(
        tmp_path, required_dates=[(7, 24)], ratio=0.2
    )[(7, 24)]

    assert info.cutoff == 80
    assert info.episode_count == 3
    assert info.expert_episode_keys == frozenset(
        stable_episode_key(episode_id) for episode_id in ("a", "b", "c")
    )


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


def test_expert_masks_align_with_existing_validation_splits(
    tmp_path: Path,
) -> None:
    build_shard(
        tmp_path / "7.19-old.cache",
        list(range(1, 101)),
        source_name="7.19.jsonl.gz",
        deck_markers=[101] * 100,
    )
    build_shard(
        tmp_path / "latest.cache",
        [101, 102, 103],
        source_name="7.24.jsonl.gz",
    )
    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    try:
        splits = dataset.build_splits(
            validation_ratio=0.5,
            validation_seed=123,
            expert_episode_keys={
                (7, 19): {
                    stable_episode_key(f"episode-{marker}")
                    for marker in range(1, 101)
                },
                (7, 24): {stable_episode_key("episode-101")},
            },
            top_deck_keys=(stable_deck_key([101] * 60),),
        )
        assert len(splits.in_distribution_expert_mask) == len(
            splits.in_distribution
        )
        assert splits.in_distribution_expert_mask.all()
        assert len(splits.latest_expert_mask) == len(splits.latest)
        np.testing.assert_array_equal(
            splits.latest_expert_mask, [True, False, False]
        )
        assert splits.in_distribution_top_deck_mask.all()
        np.testing.assert_array_equal(
            splits.latest_top_deck_mask, [True, False, False]
        )
        np.testing.assert_array_equal(
            splits.latest_expert_top_deck_mask, [True, False, False]
        )
        assert splits.in_distribution_expert_top_deck_mask.all()
    finally:
        dataset.close()


def test_top_deck_masks_preserve_configuration_order_and_expert_intersections(
    tmp_path: Path,
) -> None:
    old_markers = list(range(1, 101))
    build_shard(
        tmp_path / "old.cache",
        old_markers,
        source_name="7.19.jsonl.gz",
        deck_markers=[101 if marker % 2 else 102 for marker in old_markers],
    )
    build_shard(
        tmp_path / "7.24-latest.cache",
        [101, 102, 103],
        source_name="7.24.jsonl.gz",
        deck_markers=[101, 102, 999],
    )
    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    try:
        splits = dataset.build_splits(
            validation_ratio=0.5,
            validation_seed=123,
            expert_episode_keys={
                (7, 19): {
                    stable_episode_key(f"episode-{marker}")
                    for marker in old_markers
                },
                (7, 24): {
                    stable_episode_key(f"episode-{marker}")
                    for marker in (101, 102, 103)
                },
            },
            top_deck_keys=(
                stable_deck_key([101] * 60),
                stable_deck_key([102] * 60),
            ),
        )

        assert len(splits.in_distribution_top_deck_masks) == 2
        assert len(splits.in_distribution_expert_top_deck_masks) == 2
        assert len(splits.latest_top_deck_masks) == 2
        assert len(splits.latest_expert_top_deck_masks) == 2
        np.testing.assert_array_equal(
            splits.latest_top_deck_masks[0], [True, False, False]
        )
        np.testing.assert_array_equal(
            splits.latest_top_deck_masks[1], [False, True, False]
        )
        for deck_mask, expert_deck_mask in zip(
            splits.in_distribution_top_deck_masks,
            splits.in_distribution_expert_top_deck_masks,
        ):
            np.testing.assert_array_equal(deck_mask, expert_deck_mask)
        for deck_mask, expert_deck_mask in zip(
            splits.latest_top_deck_masks,
            splits.latest_expert_top_deck_masks,
        ):
            np.testing.assert_array_equal(deck_mask, expert_deck_mask)
    finally:
        dataset.close()


def test_isolation_precedes_latest_and_in_distribution_splits(
    tmp_path: Path,
) -> None:
    build_shard(
        tmp_path / "7.19-old.cache",
        list(range(1, 101)),
        source_name="7.19.jsonl.gz",
    )
    build_shard(
        tmp_path / "7.24-latest.cache",
        [101, 102, 103],
        source_name="7.24.jsonl.gz",
    )
    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    try:
        splits = dataset.build_splits(
            validation_ratio=0.5,
            validation_seed=123,
            isolation_episode_keys={
                "val_first": {
                    (7, 19): {stable_episode_key("episode-1")},
                    (7, 24): {stable_episode_key("episode-101")},
                },
                "val_second": {
                    (7, 19): {stable_episode_key("episode-2")},
                    (7, 24): {stable_episode_key("episode-101")},
                },
                "val_third": {
                    (7, 19): {stable_episode_key("episode-3")},
                    (7, 24): {stable_episode_key("episode-102")},
                },
            },
        )

        np.testing.assert_array_equal(
            splits.isolation,
            [0, 1, 2, 100, 101],
        )
        np.testing.assert_array_equal(
            splits.isolation_masks["val_first"],
            [True, False, False, True, False],
        )
        np.testing.assert_array_equal(
            splits.isolation_masks["val_second"],
            [False, True, False, True, False],
        )
        np.testing.assert_array_equal(
            splits.isolation_masks["val_third"],
            [False, False, True, False, True],
        )
        assert splits.isolation_union_replays == 5
        np.testing.assert_array_equal(splits.latest, [102])
        assert not set(splits.isolation) & set(splits.latest)
        assert not set(splits.isolation) & set(splits.in_distribution)
        assert not set(splits.isolation) & set(splits.train)
    finally:
        dataset.close()


def test_train_replay_sampling_is_deterministic_and_keeps_replays_together(
    tmp_path: Path,
) -> None:
    old_markers = list(range(1, 101)) * 2
    build_shard(
        tmp_path / "7.19-old.cache",
        old_markers,
        source_name="7.19.jsonl.gz",
    )
    build_shard(
        tmp_path / "7.24-latest.cache",
        [101, 102],
        source_name="7.24.jsonl.gz",
    )
    dataset = MmapFeatureDataset(tmp_path, expected_signature=SIGNATURE)
    try:
        first = dataset.build_splits(
            validation_ratio=0.2,
            validation_seed=11,
            train_replay_ratio=0.5,
            train_replay_seed=29,
        )
        second = dataset.build_splits(
            validation_ratio=0.2,
            validation_seed=11,
            train_replay_ratio=0.5,
            train_replay_seed=29,
        )
        np.testing.assert_array_equal(first.train, second.train)
        assert first.eligible_train_samples > len(first.train)
        assert first.eligible_train_replays > first.selected_train_replays

        train_ids = set(map(int, first.train))
        validation_ids = set(map(int, first.in_distribution))
        memberships: dict[int, set[str]] = {}
        for global_id in range(len(old_markers)):
            key = stable_episode_key(f"episode-{old_markers[global_id]}")
            if global_id in train_ids:
                group = "train"
            elif global_id in validation_ids:
                group = "validation"
            else:
                group = "not-selected"
            memberships.setdefault(key, set()).add(group)
        assert all(len(groups) == 1 for groups in memberships.values())
    finally:
        dataset.close()


def test_collate_rebases_options_and_pads_action_offsets(tmp_path: Path) -> None:
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
        assert batch.encoder_pokemon_appear.shape == (
            2,
            POKEMON_ENCODER_TOKENS,
        )
        assert batch.own_summary.shape == (2, OWN_SUMMARY_DIM)
        assert batch.opponent_summary.shape == (2, OPPONENT_SUMMARY_DIM)
        assert batch.global_summary.shape == (2, GLOBAL_SUMMARY_DIM)
        assert batch.option_categorical.shape == (4, OPTION_CATEGORICAL_DIM)
        assert batch.option_numeric.shape == (4, OPTION_NUMERIC_DIM)
        assert batch.action_option_offset.shape == (2 * 64 + 1,)
        assert batch.encoder_index.dtype == np.int32
        assert batch.action_option_index.dtype == np.int64
        assert batch.encoder_value.dtype == np.float16
        assert batch.option_numeric.dtype == np.float16
        assert batch.target.dtype == np.int64
        assert batch.action_count.dtype == np.int64
    finally:
        dataset.close()
