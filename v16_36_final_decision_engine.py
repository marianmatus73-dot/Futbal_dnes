
"""
V16.36 FINAL DECISION INTELLIGENCE ENGINE

Combines ensemble, risk and bankroll into final action.
"""


def final_decision(ensemble_score, risk, bankroll):
    if ensemble_score >= 0.75 and risk == "ACCEPT":
        action = "PLAY"
        stake = round(bankroll * 0.02, 2)
    else:
        action = "PASS"
        stake = 0

    return {
        "action": action,
        "stake": stake,
        "ensemble_score": ensemble_score,
        "status": "READY"
    }
