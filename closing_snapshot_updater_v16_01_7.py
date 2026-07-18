
"""
V16.01.7 Closing Snapshot Updater

Updates stored opening snapshots with closing odds.

Safe extension:
- does not modify main.py
- prepares records for CLV calculation
"""

from snapshot_storage_v16_01_5 import load_snapshots, save_snapshots


def update_closing(event_id, closing_odds):
    snapshots = load_snapshots()

    updated = 0

    for row in snapshots:
        if row.get("event_id") == event_id:
            row["closing_odds"] = closing_odds
            updated += 1

    save_snapshots(snapshots)

    return {
        "opening_records": len(snapshots),
        "closing_updated": updated,
        "status": "READY",
        "records": snapshots,
    }


if __name__ == "__main__":
    # create test opening snapshot
    save_snapshots([
        {
            "event_id": "closing_test_001",
            "opening_odds": 2.20,
            "closing_odds": None,
            "home_team": "Team A",
            "away_team": "Team B",
        }
    ])

    result = update_closing(
        "closing_test_001",
        1.90
    )

    print("=== V16.01.7 CLOSING UPDATE TEST ===")
    print(result)
