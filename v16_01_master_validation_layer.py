"""
V16.01 MASTER VALIDATION LAYER

Validates the complete V16.00 master output.
"""


def validate_master(intelligence_score, decision_score, execution_ready, risk_safe, stability):
    score = (
        intelligence_score * 0.25 +
        decision_score * 0.35 +
        (1 if execution_ready else 0) * 0.15 +
        (1 if risk_safe else 0) * 0.15 +
        (1 if stability == "STABLE" else 0) * 0.10
    )

    confidence_index = round(score * 100)

    status = "VALIDATED" if confidence_index >= 80 else "REVIEW"

    return {
        "intelligence_score": intelligence_score,
        "decision_score": decision_score,
        "execution_ready": execution_ready,
        "risk_safe": risk_safe,
        "stability": stability,
        "ai_confidence_index": confidence_index,
        "final_decision": "EXECUTE" if status == "VALIDATED" else "REVIEW",
        "system_status": status,
        "status": "READY"
    }