
"""
V16.05 CLV Integrated Main

Final integration candidate before modifying real main.py V16.00.
Keeps original main.py unchanged.
"""

from main_v16_04_clv_integrated import run_main


def run_v16_main(sport="all"):
    result = run_main(sport)

    return {
        "version": "V16.05",
        "sport": sport,
        "football_clv": result["football_clv"],
        "basketball": result["basketball"],
        "tennis": result["tennis"],
        "baseball": result["baseball"],
        "pipeline_status": result["pipeline_status"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="all")
    args = parser.parse_args()

    print("=== V16.05 CLV INTEGRATED MAIN TEST ===")
    print(run_v16_main(args.sport))
