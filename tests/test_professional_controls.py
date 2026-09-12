from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from core.config import Settings
from core.market import no_vig_probs
from core.football_candidate_optimizer_v14 import is_learning_observation_odds
from core.professional_risk import (
    _settled_profile,
    apply_professional_risk_controls,
    calibrated_probability,
    conservative_probability,
    effective_confidence,
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
        self.assertNotEqual(sport_policy("baseball"), sport_policy("handball"))
        baseball = settings_for_sport(self.settings, "baseball")
        handball = settings_for_sport(self.settings, "handball")
        self.assertLess(baseball.min_edge, handball.min_edge)
        self.assertGreater(baseball.max_stake_pct, handball.max_stake_pct)
        self.assertEqual(sport_policy("football").min_odds, 1.20)

    def test_football_balanced_learning_includes_lower_odds(self) -> None:
        self.assertTrue(is_learning_observation_odds(1.20))
        self.assertTrue(is_learning_observation_odds(1.75))
        self.assertTrue(is_learning_observation_odds(2.20))
        self.assertFalse(is_learning_observation_odds(1.19))
        self.assertFalse(is_learning_observation_odds(2.21))

    def test_calibration_uses_event_market_not_mixed_price_hit_rate(self) -> None:
        calibrated = calibrated_probability(
            .62,
            samples=500,
            hit_rate=.30,
            market_probability=.50,
        )
        self.assertGreater(calibrated, .58)
        self.assertLess(calibrated, .62)

    def test_clean_football_engine_can_publish_strong_lower_odds_candidate(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE sport_bets (sport TEXT, result TEXT, engine_version TEXT, "
                "market TEXT, odds REAL)"
            )
        bet = Bet(
            sport="football", league="L", event="Favourite vs Visitor",
            market="h2h", selection="Favourite", odds=1.50,
            prob_model=.75, prob_market=2 / 3, prob_final=.75,
            edge=.125, stake=5, bookmaker="Book",
            start_time="2026-09-10T18:00:00Z", score=80,
        )
        output = {"result": SportResult(sport="football", mode="scan", bets=[bet])}
        summary = apply_professional_risk_controls([output], self.settings)
        self.assertEqual(summary.accepted, 1)
        self.assertEqual(len(output["result"].bets), 1)
        self.assertGreaterEqual(output["result"].bets[0].edge, .04)

    def test_early_football_confidence_cannot_claim_certainty(self) -> None:
        bet = Bet(
            sport="football", league="L", event="A vs B", market="h2h",
            selection="A", odds=2.0, prob_model=.60, prob_market=.50,
            prob_final=.60, edge=.20, stake=2, bookmaker="Book",
            start_time="2026-09-20", score=100, release_stage="EARLY",
            lineup_verified=False,
        )
        self.assertEqual(effective_confidence(bet, samples=500), 75)
        bet.release_stage = "FINAL"
        bet.lineup_verified = True
        self.assertEqual(effective_confidence(bet, samples=500), 92)

    def test_football_tip_pool_is_balanced_across_odds_bands(self) -> None:
        high = [
            Bet(
                sport="football", league="L", event=f"High {index}",
                market="h2h", selection="Away", odds=3.80,
                prob_model=.29, prob_market=.263, prob_final=.29,
                edge=.102, stake=5, bookmaker="Book",
                start_time=f"2026-09-{20 + index}", score=100,
                release_stage="EARLY",
            )
            for index in range(3)
        ]
        lower = [
            Bet(
                sport="football", league="L", event=f"Lower {index}",
                market="h2h", selection="Home", odds=1.50,
                prob_model=.72, prob_market=2 / 3, prob_final=.72,
                edge=.08, stake=5, bookmaker="Book",
                start_time=f"2026-09-{25 + index}", score=80,
                release_stage="EARLY",
            )
            for index in range(2)
        ]
        output = {
            "result": SportResult(
                sport="football", mode="scan", bets=high + lower
            )
        }
        summary = apply_professional_risk_controls([output], self.settings)
        accepted = output["result"].bets
        self.assertEqual(summary.accepted, 4)
        self.assertEqual(sum(bet.odds >= 3 for bet in accepted), 2)
        self.assertEqual(sum(bet.odds < 1.60 for bet in accepted), 2)
        self.assertEqual(
            summary.rejected_reasons.get(
                "football: odds band tip limit reached"
            ),
            1,
        )

    def test_uncertainty_haircut_is_equal_in_edge_space(self) -> None:
        short_probability = .75
        long_probability = .40
        short_lower = conservative_probability(
            short_probability, 0, odds=1.50
        )
        long_lower = conservative_probability(
            long_probability, 0, odds=3.00
        )
        self.assertAlmostEqual(
            (short_probability - short_lower) * 1.50,
            .02,
        )
        self.assertAlmostEqual(
            (long_probability - long_lower) * 3.00,
            .02,
        )

    def test_football_v2_calibration_ignores_legacy_candidate_history(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE sport_bets (sport TEXT, result TEXT, engine_version TEXT, "
                "market TEXT, odds REAL)"
            )
            conn.executemany(
                "INSERT INTO sport_bets VALUES ('football', ?, '', 'h2h', 1.50)",
                [("WON",)] * 90 + [("LOST",)] * 10,
            )
            conn.executemany(
                "INSERT INTO sport_bets VALUES "
                "('football', ?, 'football-2.0', 'h2h', 1.50)",
                [("WON",)] * 2 + [("LOST",)] * 3,
            )
            conn.executemany(
                "INSERT INTO sport_bets VALUES "
                "('football', ?, 'football-2.0', 'totals_2.5', 1.50)",
                [("WON",)] * 8,
            )
            conn.executemany(
                "INSERT INTO sport_bets VALUES "
                "('football', ?, 'football-2.0', 'h2h', 3.50)",
                [("WON",)] * 7,
            )
        self.assertEqual(
            _settled_profile(self.settings, "football", "h2h", 1.50),
            (5, .40),
        )

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
            selection="A", odds=1.90, prob_model=.62, prob_market=.53,
            prob_final=.62, edge=.178, stake=30, bookmaker="Book",
            start_time="2026-08-20T20:00:00Z", score=85,
        )
        output = {"result": SportResult(sport="football", mode="scan", bets=[bet])}
        summary = apply_professional_risk_controls([output], self.settings)
        self.assertEqual(summary.accepted, 1)
        self.assertLessEqual(output["result"].bets[0].stake, 7.50)

        # Re-running the same selection may confirm/update it, but must not
        # allocate the stake a second time.
        repeated_bet = replace(bet, prob_final=.62, edge=.178, stake=30)
        output["result"].bets = [repeated_bet]
        repeated = apply_professional_risk_controls([output], self.settings)
        self.assertEqual(repeated.accepted, 1)
        self.assertEqual(repeated.daily_exposure, summary.daily_exposure)
        with sqlite3.connect(self.db_path) as conn:
            allocations = conn.execute(
                "SELECT COUNT(*), SUM(stake) FROM professional_risk_allocations"
            ).fetchone()
        self.assertEqual(allocations, (1, output["result"].bets[0].stake))

        opposite = Bet(
            sport="football", league="L", event="A vs B", market="h2h",
            selection="B", odds=2.10, prob_model=.55, prob_market=.47,
            prob_final=.55, edge=.155, stake=5, bookmaker="Book",
            start_time="2026-08-20T20:00:00Z", score=85,
        )
        output["result"].bets = [opposite]
        conflicting = apply_professional_risk_controls([output], self.settings)
        self.assertEqual(conflicting.accepted, 0)
        self.assertEqual(
            conflicting.rejected_reasons.get(
                "football: opposite selection already allocated today"
            ),
            1,
        )

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
            selection="A", odds=1.90, prob_model=.65, prob_market=.53,
            prob_final=.65, edge=.235, stake=5, bookmaker="Book",
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
            prob_model=.604, prob_market=.50, prob_final=.604, edge=.208,
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


