
"""
V16.01.6 Football Snapshot Collector

Collects and stores football market snapshots.

Safe extension:
- does not modify main.py
- uses snapshot storage layer
"""

from snapshot_storage_v16_01_5 import save_snapshots


def collect_snapshot(event):
    snapshot = {
        "event_id": event.get("event_id"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "opening_odds": event.get("opening_odds"),
        "closing_odds": event.get("closing_odds"),
        "bookmaker": event.get("bookmaker"),
        "timestamp": event.get("timestamp"),
    }

    return snapshot


def run_collector(events):
    snapshots = []

    for event in events:
        snapshot = collect_snapshot(event)

        if snapshot["event_id"]:
            snapshots.append(snapshot)

    storage = save_snapshots(snapshots)

    return {
        "version": "v16.01.6",
        "events_received": len(events),
        "snapshots_saved": storage["records_saved"],
        "storage_status": storage["status"],
    }


if __name__ == "__main__":
    test_events = [
        {
            "event_id": "collector_test_001",
            "home_team": "Team A",
            "away_team": "Team B",
            "opening_odds": 2.10,
            "closing_odds": None,
            "bookmaker": "demo",
        }
    ]

    print("=== V16.01.6 SNAPSHOT COLLECTOR TEST ===")
    print(run_collector(test_events))
