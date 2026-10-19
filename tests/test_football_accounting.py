from __future__ import annotations

import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import aiohttp  # noqa: F401
except ModuleNotFoundError:
    sys.modules["aiohttp"] = types.ModuleType("aiohttp")

from core.config import Settings
from core.football_settlement import (
    CompletedFootballGame,
    FootballSettlementEngine,
    OpenFootballBet,
    football_sport_key,
)
from core.sport_quant import init_sport_db
from core.sport_settlement import backfill_settled_profit


class FootballAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "bets.db"
        self.settings = Settings(db_file=str(self.db_path))
        init_sport_db(self.settings)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _insert_bet(self, result: str = "OPEN") -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO sport_bets (
                    sport, league, event, market, selection, odds, stake,
                    start_time, source_hash, result
                ) VALUES ('football', 'Test', 'A vs B', 'h2h', 'A', 2.5,
                          1.2, '2026-08-10T10:00:00Z', ?, ?)
                """,
                (f"hash-{result}", result),
            )
            return int(cursor.lastrowid)

    def test_football_settlement_writes_profit(self) -> None:
        bet_id = self._insert_bet()
        engine = FootballSettlementEngine(self.settings)
        bet = OpenFootballBet(
            bet_id=bet_id, source_hash="hash-OPEN", sport_key="soccer_test",
            league="Test", event="A vs B", market="h2h", selection="A",
            start_time="2026-08-10T10:00:00Z", home_team="A", away_team="B",
            odds=2.5, stake=1.2,
        )
        game = CompletedFootballGame(
            event_id="event-1", sport_key="soccer_test", home_team="A",
            away_team="B", commence_time=bet.start_time, home_goals=2,
            away_goals=0, last_update="2026-08-10T12:00:00Z",
        )

        self.assertTrue(engine._save_settlement(bet, game, "WON"))
        with sqlite3.connect(self.db_path) as conn:
            result, profit, units = conn.execute(
                "SELECT result, profit, profit_units FROM sport_bets WHERE id=?",
                (bet_id,),
            ).fetchone()
        self.assertEqual(result, "WON")
        self.assertAlmostEqual(profit, 1.8)
        self.assertAlmostEqual(units, 1.5)

    def test_totals_and_double_chance_are_settled_from_score(self) -> None:
        game = CompletedFootballGame(
            event_id="event-1", sport_key="soccer_test", home_team="A",
            away_team="B", commence_time="2026-08-10T10:00:00Z",
            home_goals=2, away_goals=1, last_update="2026-08-10T12:00:00Z",
        )
        base = dict(
            bet_id=1, source_hash="hash", sport_key="soccer_test",
            league="Test", event="A vs B", start_time=game.commence_time,
            home_team="A", away_team="B", odds=1.5, stake=1.0,
        )
        over = OpenFootballBet(**base, market="totals_2.5", selection="Over 2.5 gólu")
        under = OpenFootballBet(**base, market="totals_2.5", selection="Under 2.5 gólu")
        chance = OpenFootballBet(**base, market="double_chance", selection="1X")
        self.assertEqual(FootballSettlementEngine._settle_result(over, game), "WON")
        self.assertEqual(FootballSettlementEngine._settle_result(under, game), "LOST")
        self.assertEqual(FootballSettlementEngine._settle_result(chance, game), "WON")

    def test_display_league_resolves_to_provider_key(self) -> None:
        self.assertEqual(
            football_sport_key("UEFA Conference League"),
            "soccer_uefa_europa_conference_league",
        )
        self.assertEqual(football_sport_key("soccer_epl"), "soccer_epl")

    def test_open_totals_without_feature_row_is_still_loaded(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sport_bets (
                    sport, league, event, market, selection, odds, stake,
                    start_time, source_hash, result
                ) VALUES (
                    'football', 'UEFA Conference League', 'A vs B',
                    'totals_2.5', 'Over 2.5 gólu', 1.6, 1.0,
                    '2026-10-15T19:00:00Z', 'totals-without-feature', 'OPEN'
                )
                """
            )
        rows = FootballSettlementEngine(self.settings).load_open_bets()
        row = next(item for item in rows if item.source_hash == "totals-without-feature")
        self.assertEqual(row.sport_key, "soccer_uefa_europa_conference_league")

    def test_backfill_repairs_legacy_wins_and_losses(self) -> None:
        won_id = self._insert_bet("WON")
        lost_id = self._insert_bet("LOST")
        self.assertEqual(backfill_settled_profit(self.settings), 2)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, profit, profit_units FROM sport_bets ORDER BY id"
            ).fetchall()
        values = {row[0]: (row[1], row[2]) for row in rows}
        self.assertEqual(values[won_id], (1.8, 1.5))
        self.assertEqual(values[lost_id], (-1.2, -1.0))

    def test_stale_unmatched_is_voided_only_after_successful_fetch(self) -> None:
        bet_id = self._insert_bet()
        engine = FootballSettlementEngine(self.settings)
        bet = OpenFootballBet(
            bet_id=bet_id, source_hash="hash-OPEN", sport_key="soccer_test",
            league="Test", event="A vs B", market="h2h", selection="A",
            start_time="2026-07-01T10:00:00Z", home_team="A", away_team="B",
            odds=2.5, stake=1.2,
        )

        self.assertEqual(engine._expire_stale_unmatched([bet], set()), 0)
        self.assertEqual(
            engine._expire_stale_unmatched([bet], {"soccer_test"}),
            1,
        )
        with sqlite3.connect(self.db_path) as conn:
            result, source, profit = conn.execute(
                "SELECT result, settlement_source, profit FROM sport_bets WHERE id=?",
                (bet_id,),
            ).fetchone()
        self.assertEqual(result, "VOID")
        self.assertEqual(source, "historical_unresolved_timeout")
        self.assertEqual(profit, 0.0)


if __name__ == "__main__":
    unittest.main()
