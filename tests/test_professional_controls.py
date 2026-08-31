from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from core.config import Settings
from core.market import no_vig_probs
from core.professional_risk import (
    apply_professional_risk_controls,
    calibrated_probability,
)
from core.sport_policy import settings_for_sport, sport_policy
from core.sport_walkforward import walkforward_report
from core.sport_context import SportContextDatabase
from core.types import Bet, SportResult


class ProfessionalControlsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "bets.db"
        self.settings = Settings(bank=1000, db_file=str(self.db_path))
        os.environ.pop("BANKROLL_PEAK", None)

    def tearDown(self) -> None:
        os.environ.pop("BANKROLL_PEAK", None)
        self.temp_dir.cleanup()

    def test_no_vig_probabilities_remove_overround(self) -> None:
        probabilities = no_vig_probs({"A": 1.80, "B": 2.10})
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertNotAlmostEqual(probabilities["A"], 1 / 1.80)

    def test_each_sport_has_independent_limits(self) -> None:
        self.assertNotEqual(sport_policy("baseball"), sport_policy("mma"))
        baseball = settings_for_sport(self.settings, "baseball")
        mma = settings_for_sport(self.settings, "mma")
        self.assertLess(baseball.min_edge, mma.min_edge)
        self.assertGreater(baseball.max_stake_pct, mma.max_stake_pct)

    def test_calibration_uses_event_market_not_mixed_price_hit_rate(self) -> None:
        calibrated = calibrated_probability(
            .62,
            samples=500,
            hit_rate=.30,
            market_probability=.50,
        )
        self.assertGreater(calibrated, .58)
        self.assertLess(calibrated, .62)

    def test_risk_engine_caps_stake_and_drawdown_pauses(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE sport_bets (sport TEXT, result TEXT)"
            )
            conn.executemany(
                "INSERT INTO sport_bets VALUES ('football', ?)",
                [("WON",)] * 120 + [("LOST",)] * 80,
            )
        bet = Bet(
            sport="football", league="L", event="A vs B", market="h2h",
            selection="A", odds=1.90, prob_model=.70, prob_market=.53,
            prob_final=.70, edge=.33, stake=30, bookmaker="Book",
            start_time="2026-08-20T20:00:00Z", score=85,
        )
        output = {"result": SportResult(sport="football", mode="scan", bets=[bet])}
        summary = apply_professional_risk_controls([output], self.settings)
        self.assertEqual(summary.accepted, 1)
        self.assertLessEqual(output["result"].bets[0].stake, 7.50)

        # Re-running on the same day must not allocate the same event again.
        output["result"].bets = [bet]
        repeated = apply_professional_risk_controls([output], self.settings)
        self.assertEqual(repeated.accepted, 0)
        self.assertGreater(repeated.daily_exposure, 0)

        os.environ["BANKROLL_PEAK"] = "1200"
        bet.stake = 5
        output["result"].bets = [bet]
        paused = apply_professional_risk_controls([output], self.settings)
        self.assertTrue(paused.drawdown_paused)
        self.assertEqual(paused.accepted, 0)

    def test_walkforward_split_is_chronological(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE sport_bets (
                    id INTEGER, sport TEXT, prob_final REAL, result TEXT,
                    start_time TEXT, created_at TEXT
                )
                """
            )
            conn.executemany(
                "INSERT INTO sport_bets VALUES (?, 'baseball', .60, ?, ?, ?)",
                [
                    (index, "WON" if index % 2 else "LOST", f"2026-01-{index:02d}", "")
                    for index in range(1, 41)
                ],
            )
        report = walkforward_report(self.settings, min_samples=30)["baseball"]
        self.assertEqual(report["split"], "chronological_70_30")
        self.assertEqual(report["train_samples"], 28)
        self.assertEqual(report["test_samples"], 12)

    def test_only_verified_context_changes_risk_decision(self) -> None:
        context_db = SportContextDatabase(self.settings)
        context_db.init_db()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE sport_bets (sport TEXT, result TEXT)")
            conn.executemany(
                "INSERT INTO sport_bets VALUES ('football', ?)",
                [("WON",)] * 120 + [("LOST",)] * 80,
            )
            conn.execute(
                """
                INSERT INTO sport_context_features (
                    sport, event, injury_impact, source, captured_at, source_hash
                ) VALUES ('football', 'A vs B', .12, 'verified-provider',
                          ?, 'ctx-1')
                """,
                (datetime.now(timezone.utc).isoformat(),),
            )
        bet = Bet(
            sport="football", league="L", event="A vs B", market="h2h",
            selection="A", odds=1.90, prob_model=.70, prob_market=.53,
            prob_final=.70, edge=.33, stake=5, bookmaker="Book",
            start_time="2026-08-20T20:00:00Z", score=85,
        )
        output = {"result": SportResult(sport="football", mode="scan", bets=[bet])}
        summary = apply_professional_risk_controls([output], self.settings)
        self.assertEqual(summary.accepted, 0)

    def test_risk_engine_keeps_searching_after_rejected_longshots(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE sport_bets (sport TEXT, result TEXT)")
            conn.executemany(
                "INSERT INTO sport_bets VALUES ('football', ?)",
                [("WON",)] * 300 + [("LOST",)] * 276,
            )

        longshots = [
            Bet(
                sport="football", league="L", event=f"Long {index}",
                market="h2h", selection="Away", odds=4.40,
                prob_model=.30, prob_market=.23, prob_final=.30, edge=.32,
                stake=5, bookmaker="Book",
                start_time=f"2026-09-{index + 1:02d}", score=100,
            )
            for index in range(3)
        ]
        eligible = Bet(
            sport="football", league="L", event="Good A vs B",
            market="h2h", selection="A", odds=2.00,
            prob_model=.62, prob_market=.50, prob_final=.62, edge=.24,
            stake=5, bookmaker="Book", start_time="2026-09-10", score=85,
        )
        output = {
            "result": SportResult(
                sport="football", mode="scan", bets=longshots + [eligible]
            )
        }

        summary = apply_professional_risk_controls([output], self.settings)

        self.assertEqual(summary.accepted, 1)
        self.assertEqual(output["result"].bets[0].event, "Good A vs B")
        self.assertEqual(
            summary.rejected_reasons.get(
                "football: odds outside sport limits"
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()


