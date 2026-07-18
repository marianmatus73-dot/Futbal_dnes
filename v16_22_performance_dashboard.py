
"""
V16.22 PERFORMANCE DASHBOARD

Aggregates model performance metrics.
"""


def build_dashboard(records):
    wins = sum(1 for r in records if r.get("result") == "WIN")
    losses = sum(1 for r in records if r.get("result") == "LOSS")

    total = wins + losses
    roi = 0

    if total:
        roi = round((wins - losses) / total, 4)

    return {
        "bets": total,
        "wins": wins,
        "losses": losses,
        "roi": roi,
        "status": "READY"
    }
