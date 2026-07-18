
"""
V16.01.8 Full CLV Pipeline Runner

Combines:
- collector
- storage
- closing update
- CLV calculation

Safe extension:
- does not modify main.py
"""

from football_snapshot_collector_v16_01_6 import run_collector
from closing_snapshot_updater_v16_01_7 import update_closing
from clv_bridge_v16_01 import run_clv_bridge


def run_full_pipeline(events):
    collector_result = run_collector(events)

    event_id = events[0]["event_id"]
    closing_result = update_closing(
        event_id,
        events[0].get("closing_odds")
    )

    clv_result = run_clv_bridge(
        closing_result["records"]
    )

    return {
        "version": "v16.01.8",
        "events_received": collector_result["events_received"],
        "opening_saved": collector_result["snapshots_saved"],
        "closing_updated": closing_result["closing_updated"],
        "clv_records": clv_result["clv_records_created"],
        "pipeline_status": "READY",
        "clv_output": clv_result["records"],
    }


if __name__ == "__main__":
    test_event = [
        {
            "event_id": "full_pipeline_test_001",
            "home_team": "Team A",
            "away_team": "Team B",
            "opening_odds": 2.20,
            "closing_odds": 1.90,
            "bookmaker": "demo",
        }
    ]

    print("=== V16.01.8 FULL CLV PIPELINE TEST ===")
    print(run_full_pipeline(test_event))
