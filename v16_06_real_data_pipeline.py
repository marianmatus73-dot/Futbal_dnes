
"""
V16.06 REAL DATA PIPELINE

Full flow:
collector -> closing -> CLV bridge
"""

from v16_06_real_data_collector import collect
from v16_06_closing_processor import process_closing
from main_bridge_v16_02_1 import run_main_bridge


def run_pipeline():
    events = [
        {
            "event_id": "real_demo_001",
            "home_team": "Team A",
            "away_team": "Team B",
            "opening_odds": 2.10,
            "closing_odds": 1.90
        }
    ]

    collected = collect(events)
    closing = process_closing(events)
    clv = run_main_bridge()

    return {
        "version": "V16.06",
        "collector": collected["status"],
        "closing": closing["status"],
        "clv": clv["pipeline_status"],
        "status": "READY"
    }


if __name__ == "__main__":
    print("=== V16.06 REAL DATA PIPELINE ===")
    print(run_pipeline())
