from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.features import SparseVector, _add_known_deck_token


class KnownDeckFeatureTest(unittest.TestCase):
    def test_token_preserves_one_entry_per_physical_card(self) -> None:
        sparse = SparseVector()
        sparse.word_start()

        _add_known_deck_token(sparse, [7, 7, 9], card_count=16)

        self.assertEqual(sparse.index, [7, 7, 9])
        self.assertEqual(sparse.value, [1.0, 1.0, 1.0])
        self.assertEqual(sparse.offset, [0])
        self.assertEqual(sparse.pos, 16)

    def test_empty_token_only_advances_its_vocabulary_range(self) -> None:
        sparse = SparseVector()
        sparse.word_start()

        _add_known_deck_token(sparse, [], card_count=16)

        self.assertEqual(sparse.index, [])
        self.assertEqual(sparse.value, [])
        self.assertEqual(sparse.offset, [0])
        self.assertEqual(sparse.pos, 16)


if __name__ == "__main__":
    unittest.main()
