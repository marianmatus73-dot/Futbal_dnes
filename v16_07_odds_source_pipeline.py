
"""
V16.07 ODDS SOURCE PIPELINE

SOURCE
 -> NORMALIZER
 -> V16.06 COLLECTOR
 -> CLV
"""

from v16_07_odds_source_connector import connect
from v16_06_real_data_collector import collect
from main_bridge_v16_02_1 import run_main_bridge


def run_pipeline():
    source = [
        {
            "event_id": "source_demo_001",
            "home_team": "Team A",
            "away_team": "Team B",
            "opening_odds": 2.20,
            "current_odds": 2.00,
            "bookmaker": "demo"
        }
    ]

    normalized = connect(source)
    stored = collect(normalized["rows"])
    clv = run_main_bridge()

    return {
        "version": "V16.07",
        "source": normalized["status"],
        "collector": stored["status"],
        "clv": clv["pipeline_status"],
        "status": "READY"
    }


if __name__ == "__main__":
    print("=== V16.07 ODDS SOURCE PIPELINE ===")
    print(run_pipeline())
