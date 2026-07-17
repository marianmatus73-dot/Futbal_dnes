from __future__ import annotations

import argparse
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_football_pipeline():
    return {
        "sport": "football",
        "version": "v15.64",
        "modules": {
            "source_loader": "READY",
            "resolver": "READY",
            "opening": "READY",
            "closing": "READY",
            "clv": "READY",
            "learning": "READY",
            "feature_bridge": "READY",
            "report": "READY",
        },
    }


def run_generic_sport_pipeline(sport):
    return {
        "sport": sport,
        "framework": "READY",
        "status": "READY",
    }


def run_master_pipeline(sport_mode="all"):
    sports = ["football", "basketball", "tennis", "baseball"]

    if sport_mode != "all":
        sports = [sport_mode]

    output = {
        "version": "v15.64",
        "created_at": now(),
        "sports": {},
        "pipeline_status": "READY",
    }

    for sport in sports:
        if sport == "football":
            output["sports"][sport] = run_football_pipeline()
        else:
            output["sports"][sport] = run_generic_sport_pipeline(sport)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="all")
    args = parser.parse_args()

    print(run_master_pipeline(args.sport))
