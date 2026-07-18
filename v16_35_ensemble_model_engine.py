
"""
V16.35 ENSEMBLE MODEL ENGINE

Combines multiple model signals into one score.
"""


def combine_models(prediction, clv_score, market_score, risk_score):
    weights = {
        "prediction": 0.4,
        "clv": 0.25,
        "market": 0.20,
        "risk": 0.15
    }

    final_score = (
        prediction * weights["prediction"] +
        clv_score * weights["clv"] +
        market_score * weights["market"] +
        risk_score * weights["risk"]
    )

    return {
        "ensemble_score": round(final_score, 2),
        "models_combined": 4,
        "status": "READY"
    }
