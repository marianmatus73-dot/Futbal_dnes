from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from datetime import date
from pathlib import Path

from core.config import Settings
from core.sport_context import SportContextDatabase
from core.sportmonks import (
    SportmonksClient,
    SportmonksError,
    _explicit_lineup_confirmation,
    sync_upcoming_context,
)


class FakeSportmonksClient(SportmonksClient):
    def __init__(self) -> None:
        pass

    def fixtures_by_date(self, fixture_date: date, max_pages: int = 10):
        return [{
            "id": 123,
            "league_id": 501,
            "name": "Home vs Away",
            "starting_at": f"{fixture_date.isoformat()} 18:00:00",
            "participants": [
                {"id": 10, "name": "Home", "meta": {"location": "home"}},
                {"id": 20, "name": "Away", "meta": {"location": "away"}},
            ],
            "metadata": [{"type": {"developer_name": "LINEUP_CONFIRMED"}, "value": True}],
            "lineups": [{"type_id": 11}],
            "sidelined": [
                {"id": 1, "participant_id": 10,
                 "sideline": {"games_missed": 2, "category": "injury"}},
            ],
            "xgfixture": [{"data": {"value": 1.2}}],
        }]


class SportmonksTests(unittest.TestCase):
    def test_paid_xg_denial_falls_back_to_free_plan_fields(self) -> None:
        class FallbackClient(SportmonksClient):
            def __init__(self):
                super().__init__("test-token", include_xg=True)
                self.includes = []

            def _get(self, path, params):
                self.includes.append(params["include"])
                if "xGFixture" in params["include"]:
                    raise SportmonksError(
                        "forbidden", status=403,
                        detail="You do not have access to the 'xgfixture' include",
                    )
                return {"data": []}

        client = FallbackClient()
        self.assertEqual(client.fixtures_by_date(date(2026, 8, 20)), [])
        self.assertEqual(len(client.includes), 2)
        self.assertIn("xGFixture", client.includes[0])
        self.assertNotIn("xGFixture", client.includes[1])
        self.assertFalse(client.include_xg)

    def test_confirmation_requires_explicit_true_metadata(self) -> None:
        self.assertFalse(_explicit_lineup_confirmation([]))
        self.assertFalse(_explicit_lineup_confirmation([{"name": "lineup_confirmed", "value": False}]))
        self.assertTrue(_explicit_lineup_confirmation([
            {"type": {"developer_name": "LINEUP_CONFIRMED"}, "value": True}
        ]))

    def test_sync_persists_raw_snapshot_and_safe_context(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            db_path = Path(folder) / "bets.db"
            database = SportContextDatabase(Settings(db_file=str(db_path)))
            summary = sync_upcoming_context(
                FakeSportmonksClient(), database, start_date=date(2026, 8, 20), days=1
            )
            self.assertEqual(summary.fixtures_received, 1)
            self.assertEqual(summary.snapshots_added, 1)
            self.assertEqual(summary.confirmed_lineups, 1)
            self.assertGreater(summary.home_absence_impact, 0)
            with sqlite3.connect(db_path) as conn:
                context = conn.execute(
                    """SELECT lineup_confirmed, injury_impact, suspension_impact,
                              home_team, away_team, home_absence_impact,
                              away_absence_impact
                       FROM sport_context_features"""
                ).fetchone()
                snapshots = conn.execute("SELECT COUNT(*) FROM sport_provider_snapshots").fetchone()[0]
            self.assertEqual(context[:5], (1, 0.0, 0.0, "Home", "Away"))
            self.assertGreater(context[5], 0)
            self.assertEqual(context[6], 0)
            self.assertEqual(snapshots, 1)

    def test_odds_event_links_to_provider_by_teams_and_kickoff(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            db_path = Path(folder) / "bets.db"
            database = SportContextDatabase(Settings(db_file=str(db_path)))
            database.init_db()
            captured = datetime.now(timezone.utc).isoformat()
            with database.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sport_context_features (
                        sport, event, external_event_id, start_time,
                        home_team, away_team, home_absence_impact,
                        away_absence_impact, source, captured_at, source_hash
                    ) VALUES ('football', 'FC København vs Hearts', 'sm-10',
                              '2026-08-22 18:00:00', 'FC København', 'Hearts',
                              .012, .004, 'sportmonks-v3', ?, 'mapping-test')
                    """,
                    (captured,),
                )
            context = database.latest(
                "football", "FC Copenhagen vs Heart of Midlothian",
                "odds-99", "2026-08-22T18:30:00Z",
            )
            self.assertTrue(context.verified)
            self.assertAlmostEqual(context.home_absence_impact, .012)
            self.assertAlmostEqual(
                database.selection_availability_adjustment(context, "FC Copenhagen"),
                -.008,
            )
            with database.connect() as conn:
                link = conn.execute(
                    "SELECT provider_event_id FROM sport_event_identity_links "
                    "WHERE consumer_event_id='odds-99'"
                ).fetchone()
            self.assertEqual(link[0], "sm-10")


if __name__ == "__main__":
    unittest.main()


