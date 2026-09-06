from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

from core.config import Settings
from core.learning_observation_settlement import settle_learning_observations
from sports.handball import HandballModule


class LearningObservationSettlementTests(unittest.IsolatedAsyncioTestCase):
    async def test_settles_only_exact_external_event_id_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bets.db"
            settings = Settings(odds_api_key="test", db_file=str(database))
            HandballModule()._ensure_tables(settings)
            with closing(sqlite3.connect(database)) as conn:
                for event_id in ("right-id", "wrong-id"):
                    for selection in ("Berlin", "Draw", "Kiel"):
                        conn.execute(
                            """
                            INSERT INTO sport_learning_observations
                            (observed_at, sport, league, external_event_id, event,
                             home_team, away_team, market, selection, odds,
                             mode, result, source_hash)
                            VALUES ('2026-09-06', 'handball',
                                    'handball_germany_bundesliga', ?,
                                    'Berlin vs Kiel', 'Berlin', 'Kiel', 'h2h',
                                    ?, 2.0, 'SHADOW', 'OPEN', ?)
                            """,
                            (event_id, selection, f"{event_id}-{selection}"),
                        )
                conn.commit()
                for event_id in ("right-id", "wrong-id"):
                    conn.execute(
                        """
                        INSERT INTO sport_shadow_candidates
                        (created_at, sport, league, external_event_id, event,
                         home_team, away_team, market, selection, odds,
                         probability, model_version, result)
                        VALUES ('2026-09-06', 'handball', 'league', ?,
                                'Berlin vs Kiel', 'Berlin', 'Kiel', 'h2h',
                                'Berlin', 1.80, 0.55,
                                'market_favourite_v1', 'OPEN')
                        """,
                        (event_id,),
                    )
                conn.commit()
            scores = [{
                "id": "right-id",
                "completed": True,
                "home_team": "Berlin",
                "away_team": "Kiel",
                "scores": [
                    {"name": "Berlin", "score": "31"},
                    {"name": "Kiel", "score": "29"},
                ],
            }]
            with patch(
                "core.learning_observation_settlement.fetch_scores",
                AsyncMock(return_value=scores),
            ):
                first = await settle_learning_observations(
                    settings,
                    sport="handball",
                    sport_keys=["handball_germany_bundesliga"],
                )
                second = await settle_learning_observations(
                    settings,
                    sport="handball",
                    sport_keys=["handball_germany_bundesliga"],
                )

            self.assertEqual((first.settled_rows, first.won, first.lost), (3, 1, 2))
            self.assertEqual(second.settled_rows, 0)
            self.assertEqual(first.candidate_rows, 1)
            self.assertEqual(second.candidate_rows, 0)
            with closing(sqlite3.connect(database)) as conn:
                exact = conn.execute(
                    "SELECT selection, result, final_score FROM "
                    "sport_learning_observations WHERE external_event_id='right-id' "
                    "ORDER BY selection"
                ).fetchall()
                unmatched = conn.execute(
                    "SELECT COUNT(*) FROM sport_learning_observations "
                    "WHERE external_event_id='wrong-id' AND result='OPEN'"
                ).fetchone()[0]
                candidates = conn.execute(
                    "SELECT external_event_id, result, profit_units FROM "
                    "sport_shadow_candidates ORDER BY external_event_id"
                ).fetchall()
            self.assertEqual(exact, [
                ("Berlin", "WON", "31-29"),
                ("Draw", "LOST", "31-29"),
                ("Kiel", "LOST", "31-29"),
            ])
            self.assertEqual(unmatched, 3)
            self.assertEqual(candidates, [
                ("right-id", "WON", 0.8),
                ("wrong-id", "OPEN", None),
            ])


if __name__ == "__main__":
    unittest.main()

