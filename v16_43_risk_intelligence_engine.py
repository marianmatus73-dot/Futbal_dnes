
"""
V16.43 RISK INTELLIGENCE ENGINE

Evaluates risk before execution.
"""


def analyze_risk(exposure, bankroll, volatility, drawdown):
    risk_score = 100

    if exposure > 0.05:
        risk_score -= 25

    if volatility > 0.5:
        risk_score -= 25

    if drawdown > 0.10:
        risk_score -= 25

    if risk_score >= 75:
        decision = "SAFE"
    elif risk_score >= 50:
        decision = "REDUCE"
    else:
        decision = "BLOCK"

    return {
        "risk_score": risk_score,
        "decision": decision,
        "exposure": exposure,
        "volatility": volatility,
        "drawdown": drawdown,
        "status": "READY"
    }
