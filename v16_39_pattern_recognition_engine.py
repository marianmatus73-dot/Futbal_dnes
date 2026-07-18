
"""
V16.39 PATTERN RECOGNITION ENGINE

Detects recurring success and failure patterns.
"""


def analyze_patterns(records):
    wins = sum(1 for r in records if r.get("result") == "WIN")
    losses = sum(1 for r in records if r.get("result") == "LOSS")

    if wins > losses:
        pattern = "POSITIVE"
    elif losses > wins:
        pattern = "NEGATIVE"
    else:
        pattern = "NEUTRAL"

    return {
        "records_analyzed": len(records),
        "wins": wins,
        "losses": losses,
        "pattern": pattern,
        "status": "READY"
    }
