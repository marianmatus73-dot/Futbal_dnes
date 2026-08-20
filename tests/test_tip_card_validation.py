from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from core.tip_card_validation import TipCardValidationError, validate_tip_card


def valid_card() -> dict:
    row = {
        "sport": "baseball", "event": "A vs B", "selection": "A",
        "odds": 2.0, "model_probability": 0.60,
        "market_probability": 0.50, "edge": 0.10,
        "confidence": 80, "decision": "ACCEPT",
    }
    return {
        "schema_version": 2,
        "generated_at": datetime.now().astimezone().isoformat(),
        "publishable": True,
        "policy": {"top_limit": 5},
        "selected": [row],
        "rejected_sample": [],
    }


class TipCardValidationTests(unittest.TestCase):
    def write(self, root: Path, card: dict) -> Path:
        path = root / "latest_tip_card.json"
        path.write_text(json.dumps(card), encoding="utf-8")
        return path

    def test_accepts_complete_fresh_card(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = validate_tip_card(self.write(Path(temp), valid_card()))
            self.assertEqual(result["status"], "READY")

    def test_rejects_missing_candidate_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            card = valid_card()
            del card["selected"][0]["confidence"]
            with self.assertRaises(TipCardValidationError):
                validate_tip_card(self.write(Path(temp), card))

    def test_rejects_inconsistent_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            card = valid_card()
            card["selected"][0]["edge"] = 0.20
            with self.assertRaises(TipCardValidationError):
                validate_tip_card(self.write(Path(temp), card))


if __name__ == "__main__":
    unittest.main()
