from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.shadow_model_metrics import build_shadow_model_metrics
from sports.handball import HandballModule


class ShadowModelMetricsTests(unittest.TestCase):
    def test_repeated_snapshots_do_not_inflate_readiness(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            database = Path(directory) / "bets.db"
            settings = Settings(db_file=str(database))
            HandballModule()._ensure_tables(settings)
            with sqlite3.connect(database) as conn:
                for hour in ("10", "11"):
                    for selection, probability, result in (
                        ("Berlin", 0.55, "WON"),
                        ("Draw", 0.10, "LOST"),
                        ("Kiel", 0.35, "LOST"),
                    ):
                        conn.execute(
                            """
                            INSERT INTO sport_learning_observations
                            (observed_at, sport, league, external_event_id, event,
                             market, selection, odds, market_probability, result,
                             mode, source_hash)
                            VALUES (?, 'handball', 'league', 'event-1',
                                    'Berlin vs Kiel', 'h2h', ?, 2.0, ?, ?,
                                    'SHADOW', ?)
                            """,
                            (f"2026-09-01T{hour}:00:00Z", selection,
                             probability, result, f"{hour}-{selection}"),
                        )

            metrics = build_shadow_model_metrics(
                settings, sport="handball", minimum_events=150
            )
            self.assertEqual(metrics["settled_events"], 1)
            self.assertEqual(metrics["canonical_samples"], 3)
            self.assertEqual(metrics["raw_rows"], 6)
            self.assertEqual(metrics["duplicate_snapshots_excluded"], 3)
            self.assertEqual(metrics["probability_samples"], 3)
            self.assertEqual(metrics["readiness_pct"], 0.67)
            self.assertFalse(metrics["publishing_unlocked"])


if __name__ == "__main__":
    unittest.main()

