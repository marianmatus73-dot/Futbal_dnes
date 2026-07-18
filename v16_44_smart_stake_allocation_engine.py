
"""
V16.44 SMART STAKE ALLOCATION ENGINE

Calculates stake size from confidence and risk.
"""


def allocate_stake(bankroll, confidence, risk_decision):
    if risk_decision != "SAFE":
        stake = 0
    else:
        base_fraction = confidence * 0.03
        stake = round(bankroll * base_fraction, 2)

    return {
        "bankroll": bankroll,
        "confidence": confidence,
        "risk": risk_decision,
        "stake": stake,
        "status": "READY"
    }
