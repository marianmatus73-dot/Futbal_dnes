"""
V16.05 MASTER DECISION POLICY UPDATE ENGINE

Updates decision policy from optimized strategy state.
"""


def policy_update(strategy_score, strategy_mode, decision_threshold):
    policy_score = round(
        strategy_score * 0.7 +
        decision_threshold * 0.3,
        3
    )

    mode = "EXECUTION_READY" if policy_score >= 1.0 else "CALIBRATING"

    return {
        "strategy_score": strategy_score,
        "strategy_mode": strategy_mode,
        "decision_threshold": decision_threshold,
        "policy_score": policy_score,
        "policy_mode": mode,
        "decision_policy_updated": True,
        "policy_update_active": True,
        "status": "READY"
    }