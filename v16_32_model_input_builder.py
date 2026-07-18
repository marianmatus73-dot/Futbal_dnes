
"""
V16.32 MODEL INPUT BUILDER

Combines extracted features into model input.
"""


def build_input(features, clv=0, market_trend="STABLE"):
    return {
        "event_id": features.get("event_id"),
        "features": {
            "odds": features.get("odds_feature"),
            "source": features.get("source_feature"),
            "clv": clv,
            "market_trend": market_trend
        },
        "model_ready": True,
        "status": "READY"
    }
