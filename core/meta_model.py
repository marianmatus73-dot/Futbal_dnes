from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle


MODEL_FILE = Path("models/meta_model.pkl")


@dataclass
class MetaFeatures:
    market_probability: float
    elo_adjustment: float
    form_adjustment: float
    clv_adjustment: float
    bookmaker_grade: float
    sport_weight: float
    league_weight: float
    confidence: float
    monte_carlo_probability: float


def model_exists() -> bool:
    return MODEL_FILE.exists()


def load_model():
    if not model_exists():
        return None

    with MODEL_FILE.open("rb") as f:
        return pickle.load(f)


def ensemble_probability(features: MetaFeatures) -> float:
    p = (
        features.market_probability
        + features.elo_adjustment
        + features.form_adjustment
        + features.clv_adjustment
    )

    p *= features.bookmaker_grade
    p *= features.sport_weight
    p *= features.league_weight

    p += (features.monte_carlo_probability - p) * 0.30

    return max(0.01, min(0.99, p))


def predict_probability(features: MetaFeatures) -> float:
    if model_exists():
        model = load_model()

        values = [[
            features.market_probability,
            features.elo_adjustment,
            features.form_adjustment,
            features.clv_adjustment,
            features.bookmaker_grade,
            features.sport_weight,
            features.league_weight,
            features.confidence,
            features.monte_carlo_probability,
        ]]

        try:
            return float(model.predict_proba(values)[0][1])
        except Exception:
            pass

    return ensemble_probability(features)
