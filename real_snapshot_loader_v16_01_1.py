
"""
V16.01.1 Real Snapshot Loader

Safe extension for V16.00.
Purpose:
- prepare real football market snapshots for CLV bridge
- no changes to main.py

Expected sources can later be connected:
- sqlite database
- csv export
- existing football_market_snapshots_v14 table
"""


def normalize_snapshot(row):
    return {
        "event_id": row.get("event_id") or row.get("match_id") or row.get("id"),
        "opening_odds": row.get("opening_odds"),
        "closing_odds": row.get("closing_odds"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "bookmaker": row.get("bookmaker"),
        "timestamp": row.get("timestamp"),
    }


def load_snapshots(rows):
    normalized = []

    for row in rows:
        item = normalize_snapshot(row)

        if item["event_id"]:
            normalized.append(item)

    return {
        "version": "v16.01.1",
        "snapshots_found": len(rows),
        "valid_rows": len(normalized),
        "opening_found": sum(1 for x in normalized if x["opening_odds"]),
        "closing_found": sum(1 for x in normalized if x["closing_odds"]),
        "status": "READY",
        "snapshots": normalized,
    }


if __name__ == "__main__":
    test_rows = [
        {
            "event_id": "test_001",
            "opening_odds": 2.10,
            "closing_odds": 1.90,
            "home_team": "A",
            "away_team": "B",
            "bookmaker": "test",
        }
    ]

    result = load_snapshots(test_rows)

    print("=== V16.01.1 REAL SNAPSHOT LOADER TEST ===")
    print(result)
