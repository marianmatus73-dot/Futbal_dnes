
"""
V16.05 CLV FINAL MAIN

Final CLV integration layer for V16.00.
Keeps original main.py structure untouched.
Adds CLV hook through tested bridge.
"""

from main_bridge_v16_02_1 import run_main_bridge


def run_main(sport="all"):
    clv_result = run_main_bridge()

    return {
        "version": "V16.05_FINAL",
        "sport": sport,
        "football_clv": clv_result["football_clv"],
        "basketball": clv_result["basketball"],
        "tennis": clv_result["tennis"],
        "baseball": clv_result["baseball"],
        "pipeline_status": clv_result["pipeline_status"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="all")
    args = parser.parse_args()

    print(run_main(args.sport))
