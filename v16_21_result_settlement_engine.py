
"""
V16.21 RESULT SETTLEMENT ENGINE

Processes bet results and calculates profit/loss.
"""


def settle_bet(action, stake, odds, result):
    if action != "BET":
        profit = 0
        status = "NO_BET"

    elif result == "WIN":
        profit = round(stake * (odds - 1), 2)
        status = "WIN"

    elif result == "LOSS":
        profit = -stake
        status = "LOSS"

    else:
        profit = 0
        status = "PENDING"

    return {
        "status": status,
        "stake": stake,
        "odds": odds,
        "profit": profit
    }
