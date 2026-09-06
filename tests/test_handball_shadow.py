from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

from core.config import Settings
from sports.handball import HandballModule


class HandballShadowTests(unittest.IsolatedAsyncioTestCase):
    async def test_collects_learning_observations_without_publishing_bets(self) -> None:
        event = {
            "id": "hb-event-1",
            "home_team": "Berlin",
            "away_team": "Kiel",
            "commence_time": "2026-09-12T18:00:00Z",
            "bookmakers": [
                {
                    "title": "Book A",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Berlin", "price": 1.80},
                            {"name": "Draw", "price": 8.00},
                            {"name": "Kiel", "price": 2.20},
                        ],
                    }],
                },
                {
                    "title": "Book B",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Berlin", "price": 1.85},
                            {"name": "Draw", "price": 8.50},
                            {"name": "Kiel", "price": 2.25},
                        ],
                    }],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bets.db"
            settings = Settings(odds_api_key="test", db_file=str(database))
            with (
                patch(
                    "sports.handball.discover_active_sport_keys",
                    AsyncMock(return_value={"handball_germany_bundesliga"}),
                ),
                patch("sports.handball.fetch_odds", AsyncMock(return_value=[event])),
                patch(
                    "sports.handball.settle_learning_observations",
                    AsyncMock(),
                ) as settlement,
            ):
                from core.learning_observation_settlement import ObservationSettlementSummary
                settlement.return_value = ObservationSettlementSummary()
                result = await HandballModule().scan(settings)

            self.assertEqual(result.mode, "shadow")
            self.assertEqual(result.bets, [])
            with closing(sqlite3.connect(database)) as conn:
                observations = conn.execute(
                    "SELECT COUNT(*) FROM sport_learning_observations "
                    "WHERE sport='handball' AND mode='SHADOW'"
                ).fetchone()[0]
                bets = conn.execute(
                    "SELECT COUNT(*) FROM sport_bets WHERE sport='handball'"
                ).fetchone()[0]
                candidate = conn.execute(
                    "SELECT selection, model_version, result FROM "
                    "sport_shadow_candidates"
                ).fetchone()
            self.assertEqual(observations, 3)
            self.assertEqual(bets, 0)
            self.assertEqual(candidate, ("Berlin", "market_favourite_v1", "OPEN"))


if __name__ == "__main__":
    unittest.main()

