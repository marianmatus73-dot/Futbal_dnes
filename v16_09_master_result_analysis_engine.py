"""
V16.09 MASTER RESULT ANALYSIS ENGINE

Analyzes execution outcomes and generates intelligence feedback.
"""


def result_analysis(outcome, execution_score, profit):
    performance_score = round(
        (execution_score * 0.6) +
        (1.0 if outcome == "WIN" else 0.0) * 0.3 +
        (1.0 if profit > 0 else 0.0) * 0.1,
        3
    )

    impact = "POSITIVE" if performance_score >= 0.7 else "NEGATIVE"

    return {
        "outcome": outcome,
        "execution_score": execution_score,
        "profit": profit,
        "performance_score": performance_score,
        "strategy_impact": impact,
        "learning_value_generated": True,
        "result_analysis_active": True,
        "status": "READY"
    }