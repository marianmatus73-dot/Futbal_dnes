"""
V16.54 LIVE DECISION FUSION ENGINE

Combines live signal, prediction and risk into final action.
"""


def fuse_decision(signal, confidence, risk):
    if signal == "VALUE" and confidence >= 0.75 and risk == "SAFE":
        action = "PLAY"
    else:
        action = "PASS"

    return {
        "signal": signal,
        "confidence": confidence,
        "risk": risk,
        "action": action,
        "fusion_ready": True,
        "status": "READY"
    }