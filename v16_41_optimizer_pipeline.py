
"""
V16.41 MULTI-MARKET OPTIMIZER PIPELINE
"""

from v16_41_multi_market_optimizer import optimize_strategy


def run_pipeline():
    result = optimize_strategy(
        context="football",
        current_weight=1.05
    )

    return {
        "version": "V16.41",
        "optimizer": result,
        "status": "READY"
    }


if __name__ == "__main__":
    print("=== V16.41 OPTIMIZER PIPELINE ===")
    print(run_pipeline())
