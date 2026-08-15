from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.feature_cache import (
    ATTACK_DYNAMIC_DIM,
    ENCODER_WORDS,
    GLOBAL_SUMMARY_DIM,
    HISTORY_STEPS,
    OPPONENT_SUMMARY_DIM,
    OPTION_CATEGORICAL_DIM,
    OPTION_NUMERIC_DIM,
    OWN_SUMMARY_DIM,
    PLAYER_RESULT_WIN,
    POKEMON_DYNAMIC_DIM,
    POKEMON_ENCODER_TOKENS,
    FeatureRecord,
    PackedShard,
    PackedShardWriter,
    stable_deck_key,
    stable_episode_key,
)


def _record() -> FeatureRecord:
    return FeatureRecord(
        encoder_index=[741],
        encoder_value=[1.0],
        encoder_offset=[0] * ENCODER_WORDS,
        encoder_pokemon_appear=[0] * POKEMON_ENCODER_TOKENS,
        revealed_hand_present=[1, 0],
        own_summary=[0.0] * OWN_SUMMARY_DIM,
        opponent_summary=[0.0] * OPPONENT_SUMMARY_DIM,
        global_summary=[0.0] * GLOBAL_SUMMARY_DIM,
        option_categorical=[0] * OPTION_CATEGORICAL_DIM,
        option_numeric=[0.0] * OPTION_NUMERIC_DIM,
        pokemon_dynamic=[0.0] * POKEMON_DYNAMIC_DIM,
        attack_dynamic=[0.0] * ATTACK_DYNAMIC_DIM,
        action_option_index=[0],
        action_option_offset=[0, 1],
        action_eligible=[1],
        target=0,
        action_count=1,
        episode_key=stable_episode_key("episode"),
        deck_key=stable_deck_key([7] * 60),
        history_select_type=[0] * HISTORY_STEPS,
        history_select_context=[0] * HISTORY_STEPS,
        history_valid=[0] * HISTORY_STEPS,
        history_option_categorical=[],
        history_structural=[],
        history_pokemon_dynamic=[],
        history_attack_dynamic=[],
        history_option_offset=[0] * (HISTORY_STEPS + 1),
        player_result=PLAYER_RESULT_WIN,
    )


class RevealedHandCacheTests(unittest.TestCase):
    def test_presence_flags_round_trip_through_packed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "part.cache"
            signature = {"encoder_tokens": ENCODER_WORDS}
            writer = PackedShardWriter(path, signature, {"name": "7.1"})
            writer.add(_record())
            writer.finalize()

            shard = PackedShard(path, expected_signature=signature)
            try:
                sample = shard.sample(0)
                self.assertEqual(sample.encoder_offset.shape, (28,))
                self.assertEqual(
                    sample.revealed_hand_present.tolist(), [1, 0]
                )
            finally:
                shard.close()


if __name__ == "__main__":
    unittest.main()
