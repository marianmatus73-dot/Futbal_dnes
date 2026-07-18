
"""
V16.01.3 Real Source Reader

Reads real football_market_snapshots_v14 style data.

Safe extension:
- does not modify main.py
- accepts csv/sqlite/exported rows later
"""


def read_source(rows):
    result = []

    for row in rows:
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
            result.append(item)

    return {
        "version": "v16.01.3",
        "source": "football_market_snapshots_v14",
        "rows_found": len(rows),
        "valid_rows": len(result),
        "opening_found": sum(1 for x in result if x["opening_odds"]),
        "closing_found": sum(1 for x in result if x["closing_odds"]),
        "status": "READY",
        "rows": result,
    }


if __name__ == "__main__":
    demo_rows = [
        {
            "event_id": "real_test_001",
            "opening_odds": 2.20,
            "closing_odds": 2.00,
            "home_team": "Team A",
            "away_team": "Team B",
            "bookmaker": "demo"
        }
    ]

    print("=== V16.01.3 REAL SOURCE TEST ===")
    print(read_source(demo_rows))
