from __future__ import annotations

import unittest

from core.history_sync import HISTORY_SYNC_MAP, export_pairs, restore_pairs


class HistorySyncTests(unittest.TestCase):
    def test_restore_and_export_are_exact_inverses(self) -> None:
        self.assertEqual(dict(export_pairs()), HISTORY_SYNC_MAP)
        self.assertEqual(
            {(table, path) for path, table in restore_pairs()},
            set(HISTORY_SYNC_MAP.items()),
        )


if __name__ == "__main__":
    unittest.main()

