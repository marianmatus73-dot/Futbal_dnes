
"""
V16.47 PERFORMANCE ANALYTICS ENGINE

Calculates system performance metrics.
"""


def analyze_performance(records):
    bets = len(records)
    wins = sum(1 for r in records if r.get("result") == "WIN")
    losses = sum(1 for r in records if r.get("result") == "LOSS")

    profit = round(sum(r.get("profit", 0) for r in records), 2)

    win_rate = round(wins / bets, 2) if bets else 0

    return {
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit": profit,
        "performance_score": round(win_rate * 100, 2),
        "status": "READY"
    }
