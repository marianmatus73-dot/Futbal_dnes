
"""
V16.11 BETTING DECISION ENGINE

Combines prediction and risk into final decision.
"""


def make_decision(signal, confidence, risk):
    if signal == "VALUE" and confidence >= 0.75 and risk == "ACCEPT":
        decision = "PLAY"
    else:
        decision = "PASS"

    return {
        "signal": signal,
        "confidence": confidence,
        "risk": risk,
        "decision": decision
    }
