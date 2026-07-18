
"""
V16.03 CLV Integrated Main Wrapper

Safe integration wrapper for V16.00 main.py.
Keeps original main.py untouched and exposes CLV bridge.
"""

import argparse

from main_bridge_v16_02_1 import run_main_bridge


def run_integrated_main(sport="all"):
    result = {
        "version": "V16.03",
        "sport": sport,
        "clv_bridge": run_main_bridge(),
        "status": "READY",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="all")
    args = parser.parse_args()

    print("=== V16.03 CLV INTEGRATED MAIN TEST ===")
    print(run_integrated_main(args.sport))
