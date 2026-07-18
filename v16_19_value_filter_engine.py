
"""
V16.19 VALUE FILTER ENGINE

Combines market factors into quality score.
"""


def calculate_quality(clv, confidence, risk, anomaly):
    score = 0

    if clv > 0:
        score += 1

    if confidence >= 0.75:
        score += 1

    if risk == "ACCEPT":
        score += 1

    if anomaly:
        score -= 1

    decision = "PLAY" if score >= 2 else "PASS"

    return {
        "quality_score": score,
        "decision": decision,
        "status": "READY"
    }
