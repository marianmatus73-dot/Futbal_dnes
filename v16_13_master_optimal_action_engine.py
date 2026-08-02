"""
V16.13 MASTER OPTIMAL ACTION ENGINE

Selects the optimal action from forecasted scenarios.
"""


def select_optimal_action(best_scenario, confidence, risk_level, reward_score):
    action_score = round(
        confidence * 0.35 +
        (1 - risk_level) * 0.25 +
        reward_score * 0.25 +
        (1.0 if best_scenario == "positive" else 0.0) * 0.15,
        3
    )

    optimal_action = "EXECUTE" if action_score >= 0.80 else "WAIT"

    return {
        "best_scenario": best_scenario,
        "confidence": confidence,
        "risk_level": risk_level,
        "reward_score": reward_score,
        "action_score": action_score,
        "optimal_action": optimal_action,
        "action_selection_active": True,
        "status": "READY"
    }