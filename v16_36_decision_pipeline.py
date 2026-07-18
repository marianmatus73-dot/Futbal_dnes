
"""
V16.36 FINAL DECISION PIPELINE
"""

from v16_36_final_decision_engine import final_decision


def run_pipeline():
    result = final_decision(
        ensemble_score=0.79,
        risk="ACCEPT",
        bankroll=1000
    )

    return {
        "version": "V16.36",
        "decision": result,
        "status": "READY"
    }


if __name__ == "__main__":
    print("=== V16.36 DECISION PIPELINE ===")
    print(run_pipeline())
