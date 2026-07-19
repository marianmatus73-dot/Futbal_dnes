"""
V16.58 SELF-OPTIMIZING STRATEGY ENGINE

Optimizes strategy parameters from learning updates.
"""


def optimize_strategy(learning_weight, strategy_weight):
    if learning_weight > 1.15:
        adjustment = 0.05
    elif learning_weight < 0.90:
        adjustment = -0.05
    else:
        adjustment = 0

    new_strategy = round(max(0.0, min(strategy_weight + adjustment, 2.0)), 2)

    return {
        "learning_weight": learning_weight,
        "old_strategy_weight": strategy_weight,
        "new_strategy_weight": new_strategy,
        "strategy_optimized": True,
        "status": "READY"
    }