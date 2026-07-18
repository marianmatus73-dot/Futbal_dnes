
"""
V16.09 SIGNAL ENGINE

Creates decision signals from learned market data.
"""

from v16_08_market_database import load_history


def generate_signals():
    history = load_history()
    signals = []

    for row in history:
        clv = row.get("clv", 0)

        signal = "VALUE" if clv < 0 else "NO_VALUE"

        signals.append({
            "event_id": row.get("event_id"),
            "clv": clv,
            "signal": signal
        })

    return {
        "version": "V16.09",
        "records": len(signals),
        "signals": signals,
        "status": "READY"
    }
