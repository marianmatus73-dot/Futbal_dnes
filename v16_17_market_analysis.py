
"""
V16.17 ADVANCED MARKET ANALYSIS

Analyses odds movement and market behaviour.
"""


def analyze_market(opening, current, closing=None):
    movement = current - opening

    if movement < 0:
        trend = "SHORTENING"
    elif movement > 0:
        trend = "DRIFTING"
    else:
        trend = "STABLE"

    return {
        "opening": opening,
        "current": current,
        "closing": closing,
        "movement": movement,
        "trend": trend,
        "status": "READY"
    }
