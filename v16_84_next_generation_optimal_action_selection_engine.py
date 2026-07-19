"""
V16.84 NEXT GENERATION OPTIMAL ACTION SELECTION ENGINE

Selects best action from scenario evaluation.
"""


def select_optimal_action(scenarios, risk_level):
    actions = {
        "EXECUTE": round(scenarios["positive"] * (1 - risk_level), 3),
        "WAIT": round(scenarios["neutral"] * 0.8, 3),
        "AVOID": round(scenarios["negative"] * risk_level, 3)
    }

    best_action = max(actions, key=actions.get)

    return {
        "actions": actions,
        "best_action": best_action,
        "risk_level": risk_level,
        "reward_analysis_active": True,
        "optimal_selection_active": True,
        "status": "READY"
    }