
"""
V16.20 FINAL EXECUTION ENGINE

Converts approved decisions into execution plan.
"""


def execute(decision, quality_score, bankroll):
    if decision == "PLAY" and quality_score >= 2:
        stake = round(bankroll * 0.02, 2)
        action = "BET"
    else:
        stake = 0
        action = "PASS"

    return {
        "action": action,
        "stake": stake,
        "status": "READY"
    }
