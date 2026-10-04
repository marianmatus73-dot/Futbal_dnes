from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.pro_tipper import build_pro_tip, filter_value_tips
from core.tip_card import (
    build_low_odds_watch,
    save_latest_rejected_candidates,
    save_latest_tip_card,
)


class TipCardTests(unittest.TestCase):
    def test_low_odds_watch_is_separate_and_deduplicates_opposite_sides(self) -> None:
        accepted = build_pro_tip(
            sport="football", league="test", match="A vs B", pick="A",
            odds=1.50, model_probability=0.72, model_score=78,
        )
        rejected = {
            "sport": "football", "league": "test", "event": "A vs B",
            "selection": "B", "market": "h2h", "odds": 1.55,
            "prob_final": 0.67, "prob_market": 0.64, "score": 70,
            "rejection_reason": "conservative edge below sport minimum",
        }
        rows = build_low_odds_watch([accepted], [rejected])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pick"], "A")
        self.assertEqual(rows[0]["watch_status"], "PASSED_PRO_FILTER")

    def test_low_odds_watch_keeps_rejected_observation_without_publishing(self) -> None:
        rows = build_low_odds_watch([], [{
            "sport": "football", "league": "test", "event": "C vs D",
            "selection": "C", "market": "h2h", "odds": 1.40,
            "prob_final": 0.74, "prob_market": 0.71, "score": 68,
            "rejection_reason": "confidence below sport minimum",
        }])
        self.assertEqual(rows[0]["decision"], "REJECT")
        self.assertEqual(rows[0]["watch_status"], "WATCH_ONLY")

    def test_value_filter_uses_expected_return_for_lower_odds(self) -> None:
        tip = build_pro_tip(
            sport="football",
            league="test",
            match="A vs B",
            pick="A",
            odds=1.50,
            model_probability=0.70,
            model_score=75,
        )
        self.assertLess(tip.edge, 0.04)
        self.assertAlmostEqual(tip.model_probability * tip.odds - 1.0, 0.05)
        self.assertEqual(filter_value_tips([tip]), [tip])

    def test_card_contains_complete_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tip = build_pro_tip(
                sport="football",
                league="test",
                match="A vs B",
                pick="A",
                odds=2.0,
                model_probability=0.60,
            )
            path = save_latest_tip_card(
                [tip], [tip], export_dir=Path(temp), top_limit=5
            )
            card = json.loads(path.read_text(encoding="utf-8"))
            for candidate in card["selected"] + card["rejected_sample"]:
                for field in (
                    "sport", "event", "selection", "odds",
                    "model_probability", "market_probability", "edge",
                    "confidence",
                    "bookmaker_weight", "bookmaker_samples", "bookmaker_label",
                ):
                    self.assertNotIn(candidate.get(field), (None, ""))

    def test_empty_run_replaces_stale_card(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            export_dir = Path(temp)
            stale = export_dir / "latest_tip_card.json"
            stale.write_text('{"selected":[{"stale":true}]}', encoding="utf-8")
            save_latest_tip_card([], [], export_dir=export_dir, top_limit=5)
            card = json.loads(stale.read_text(encoding="utf-8"))
            self.assertFalse(card["publishable"])
            self.assertEqual(card["selected"], [])
            self.assertEqual(card["rejected_sample"], [])

    def test_complete_rejected_export_keeps_risk_and_selection_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tip = build_pro_tip(
                sport="football", league="test", match="A vs B",
                pick="A", odds=2.0, model_probability=0.60,
            )
            path = save_latest_rejected_candidates(
                [{"sport": "tennis", "event": "C vs D",
                  "rejection_reason": "confidence below sport minimum"}],
                [tip],
                export_dir=Path(temp),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["total"], 2)
            self.assertEqual(len(payload["candidates"]), 2)
            self.assertEqual(
                payload["candidates"][1]["rejection_stage"],
                "VALUE_OR_TOP_SELECTION",
            )


if __name__ == "__main__":
    unittest.main()

