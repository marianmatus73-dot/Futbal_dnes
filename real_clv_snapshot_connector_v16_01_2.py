
"""
V16.01.2 Real CLV Snapshot Connector

Safe extension:
- does not modify main.py
- prepares football_market_snapshots_v14 data for CLV bridge

The loader accepts:
- sqlite rows
- csv rows
- exported dictionaries
"""


def connect_snapshots(source_rows):
    normalized = []

    for row in source_rows:
        item = {
            "event_id": row.get("event_id") or row.get("match_id") or row.get("id"),
            "opening_odds": row.get("opening_odds"),
            "closing_odds": row.get("closing_odds"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "bookmaker": row.get("bookmaker"),
            "timestamp": row.get("timestamp"),
        }

        if item["event_id"]:
            normalized.append(item)

    return {
        "version": "v16.01.2",
        "source": "football_market_snapshots_v14",
        "snapshots_found": len(source_rows),
        "valid_rows": len(normalized),
        "opening_found": sum(1 for x in normalized if x["opening_odds"]),
        "closing_found": sum(1 for x in normalized if x["closing_odds"]),
        "status": "READY",
        "snapshots": normalized,
    }


if __name__ == "__main__":
    demo = [
        {
            "event_id": "demo_001",
            "opening_odds": 2.0,
            "closing_odds": 1.8,
            "home_team": "A",
            "away_team": "B",
            "bookmaker": "demo"
        }
    ]

    print("=== V16.01.2 REAL CLV DATA TEST ===")
    print(connect_snapshots(demo))
