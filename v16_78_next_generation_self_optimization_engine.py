"""
V16.78 NEXT GENERATION SELF-OPTIMIZATION ENGINE

Improves system parameters from performance analysis.
"""


def optimize_system(performance_score, health_score, current_weight):
    optimization_score = round(
        (performance_score * 0.6) +
        (health_score * 0.4),
        3
    )

    new_weight = round(
        current_weight + (optimization_score * 0.05),
        3
    )

    return {
        "performance_score": performance_score,
        "health_score": health_score,
        "optimization_score": optimization_score,
        "old_weight": current_weight,
        "new_weight": new_weight,
        "optimization_active": True,
        "status": "READY"
    }