from __future__ import annotations

import os
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


def test_missing_model_uses_concise_fallback(tmp_path: Path):
    clear_football_meta_cache()
    os.environ.pop("FOOTBALL_META_VERBOSE_REASON", None)

    prediction = predict_football_probability(
        make_features(),
        model_path=str(tmp_path / "missing.pkl"),
        metadata_path=str(tmp_path / "missing.json"),
    )

    assert prediction.model_loaded is False
    assert prediction.source == "FOOTBALL_V13_FALLBACK"
    assert "model not found" not in prediction.reason
    assert "fallback active" in prediction.reason


def test_verbose_fallback_can_be_enabled(tmp_path: Path):
    clear_football_meta_cache()
    os.environ["FOOTBALL_META_VERBOSE_REASON"] = "1"

    try:
        prediction = predict_football_probability(
            make_features(),
            model_path=str(tmp_path / "missing.pkl"),
            metadata_path=str(tmp_path / "missing.json"),
        )

        assert "model not found" in prediction.reason
    finally:
        os.environ.pop("FOOTBALL_META_VERBOSE_REASON", None)
        clear_football_meta_cache()
