from __future__ import annotations

import gzip
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from main import export_table_to_csv, import_csv_to_table


class HistoryCompressionTests(unittest.TestCase):
    def test_gzip_history_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            source_db = root / "source.db"
            target_db = root / "target.db"
            archive = root / "history_football_market_closing.csv.gz"

            for database in (source_db, target_db):
                with sqlite3.connect(database) as conn:
                    conn.execute(
                        "CREATE TABLE football_market_closing "
                        "(id INTEGER PRIMARY KEY, event TEXT, odds REAL)"
                    )
            with sqlite3.connect(source_db) as conn:
                conn.executemany(
                    "INSERT INTO football_market_closing VALUES (?, ?, ?)",
                    [(1, "A vs B", 1.80), (2, "C vs D", 2.10)],
                )

            exported = export_table_to_csv(
                Settings(db_file=str(source_db)),
                "football_market_closing",
                str(archive),
            )
            self.assertEqual(exported, 2)
            with gzip.open(archive, "rt", encoding="utf-8") as handle:
                self.assertIn("A vs B", handle.read())

            imported = import_csv_to_table(
                Settings(db_file=str(target_db)),
                "football_market_closing",
                str(archive),
            )
            self.assertEqual(imported, 2)


if __name__ == "__main__":
    unittest.main()
