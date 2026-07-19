"""
V16.53 REAL-TIME SIGNAL ENGINE

Creates live signals from market inputs.
"""


def generate_signal(clv, market_movement, value_score):
    if clv > 0 and market_movement == "SHORTENING" and value_score >= 0.7:
        signal = "VALUE"
    else:
        signal = "PASS"

    return {
        "clv": clv,
        "market_movement": market_movement,
        "value_score": value_score,
        "signal": signal,
        "signal_ready": True,
        "status": "READY"
    }