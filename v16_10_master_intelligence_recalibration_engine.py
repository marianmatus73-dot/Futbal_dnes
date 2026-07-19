"""
V16.10 MASTER INTELLIGENCE RECALIBRATION ENGINE

Recalibrates confidence, weights and thresholds from analysis results.
"""


def intelligence_recalibration(performance_score, model_weight, strategy_weight, confidence):
    adjustment = round(performance_score * 0.05, 3)

    new_model_weight = round(model_weight + adjustment, 3)
    new_strategy_weight = round(strategy_weight + adjustment, 3)
    new_confidence = round(min(confidence + adjustment, 1.0), 3)

    return {
        "performance_score": performance_score,
        "old_model_weight": model_weight,
        "new_model_weight": new_model_weight,
        "old_strategy_weight": strategy_weight,
        "new_strategy_weight": new_strategy_weight,
        "old_confidence": confidence,
        "new_confidence": new_confidence,
        "recalibration_active": True,
        "intelligence_updated": True,
        "status": "READY"
    }