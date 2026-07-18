
"""
V16.41 MULTI-MARKET STRATEGY OPTIMIZER

Optimizes strategy weights by market context.
"""


def optimize_strategy(context, current_weight=1.05):
    adjustments = {
        "football": 0.05,
        "basketball": 0.03,
        "tennis": 0.02
    }

    change = adjustments.get(context, 0)

    new_weight = max(0.0, min(current_weight + change, 2.0))

    return {
        "context": context,
        "old_weight": current_weight,
        "new_weight": round(new_weight, 2),
        "optimized": True,
        "status": "READY"
    }
