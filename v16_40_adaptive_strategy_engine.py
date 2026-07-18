
"""
V16.40 ADAPTIVE STRATEGY ENGINE

Adjusts strategy weights based on detected patterns.
"""


def adapt_strategy(pattern, current_weight=1.0):
    if pattern == "POSITIVE":
        new_weight = current_weight + 0.05
    elif pattern == "NEGATIVE":
        new_weight = current_weight - 0.05
    else:
        new_weight = current_weight

    new_weight = max(0.0, min(new_weight, 2.0))

    return {
        "old_weight": current_weight,
        "new_weight": round(new_weight, 2),
        "strategy_updated": True,
        "status": "READY"
    }
