
"""
V16.42 REAL-TIME ADAPTIVE DECISION LAYER

Adjusts decision based on live market context.
"""


def adaptive_decision(strategy_weight, market_signal):
    adjustment = 0

    if market_signal == "POSITIVE":
        adjustment = 0.05
    elif market_signal == "NEGATIVE":
        adjustment = -0.05

    final_weight = round(strategy_weight + adjustment, 2)

    return {
        "strategy_weight": strategy_weight,
        "market_signal": market_signal,
        "final_weight": final_weight,
        "decision_adapted": True,
        "status": "READY"
    }
