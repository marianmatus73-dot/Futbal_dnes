
"""
V16.04 CLV Integrated Main

Safe production candidate wrapper.
Keeps original V16.00 main.py unchanged.
"""

from main_bridge_v16_02_1 import run_main_bridge


def run_main(sport="all"):
    clv = run_main_bridge()

    return {
        "version": "V16.04",
        "sport": sport,
        "football_clv": clv["football_clv"],
        "basketball": clv["basketball"],
        "tennis": clv["tennis"],
        "baseball": clv["baseball"],
        "pipeline_status": clv["pipeline_status"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="all")
    args = parser.parse_args()

    print("=== V16.04 CLV INTEGRATED MAIN TEST ===")
    print(run_main(args.sport))
