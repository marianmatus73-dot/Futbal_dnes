from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.football_meta import (
    clear_football_meta_cache,
    predict_football_probability,
)


def make_features():
    return SimpleNamespace(
        model_consensus_probability=0.31,
        market_selection_probability=0.29,
        reliability_input=0.0,
        confidence_input=0.45,
        model_dispersion=0.02,
        market_overround=0.06,
    )


class FootballMetaFallbackTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("FOOTBALL_META_VERBOSE_REASON", None)
        clear_football_meta_cache()

    def test_missing_model_uses_concise_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            prediction = predict_football_probability(
                make_features(),
                model_path=str(tmp_path / "missing.pkl"),
                metadata_path=str(tmp_path / "missing.json"),
            )
        self.assertFalse(prediction.model_loaded)
        self.assertEqual(prediction.source, "FOOTBALL_V13_FALLBACK")
        self.assertNotIn("model not found", prediction.reason)
        self.assertIn("fallback active", prediction.reason)

    def test_verbose_fallback_can_be_enabled(self) -> None:
        os.environ["FOOTBALL_META_VERBOSE_REASON"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            prediction = predict_football_probability(
                make_features(),
                model_path=str(tmp_path / "missing.pkl"),
                metadata_path=str(tmp_path / "missing.json"),
            )
        self.assertIn("model not found", prediction.reason)


if __name__ == "__main__":
    unittest.main()

