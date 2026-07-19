"""
V16.06 MASTER AUTONOMOUS DECISION SYNC ENGINE

Synchronizes policy, confidence, risk and execution readiness.
"""


def decision_sync(policy_score, confidence, risk_safe, execution_ready):
    sync_score = round(
        (policy_score * 0.45) +
        (confidence * 0.35) +
        (1 if risk_safe else 0) * 0.10 +
        (1 if execution_ready else 0) * 0.10,
        3
    )

    decision = "EXECUTE" if sync_score >= 0.8 else "REVIEW"

    return {
        "policy_score": policy_score,
        "confidence": confidence,
        "risk_safe": risk_safe,
        "execution_ready": execution_ready,
        "sync_score": sync_score,
        "final_action": decision,
        "decision_sync_active": True,
        "status": "READY"
    }