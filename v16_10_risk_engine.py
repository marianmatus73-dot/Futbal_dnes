
"""
V16.10 RISK ENGINE

Controls exposure based on confidence.
"""


def assess_risk(confidence):
    if confidence >= 0.75:
        risk = "ACCEPT"

    elif confidence >= 0.50:
        risk = "CAUTION"

    else:
        risk = "REJECT"

    return {
        "risk": risk,
        "confidence": confidence
    }
