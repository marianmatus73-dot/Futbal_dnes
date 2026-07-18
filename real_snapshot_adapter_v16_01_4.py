
"""
V16.01.4 Real Snapshot Adapter

Purpose:
- final adapter before connecting CLV bridge to V16.00
- accepts real exported snapshot rows
- keeps V16.00 untouched
"""

def adapt_snapshot_source(rows):
    adapted = []

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
            adapted.append(item)

    return {
        "version": "v16.01.4",
        "source": "football_market_snapshots_v14",
        "rows_found": len(rows),
        "valid_rows": len(adapted),
        "opening_found": sum(1 for x in adapted if x["opening_odds"] is not None),
        "closing_found": sum(1 for x in adapted if x["closing_odds"] is not None),
        "status": "READY",
        "rows": adapted,
    }


if __name__ == "__main__":
    demo = [
        {
            "event_id": "adapter_test_001",
            "opening_odds": 2.15,
            "closing_odds": 1.95,
            "home_team": "Home",
            "away_team": "Away",
            "bookmaker": "demo"
        }
    ]

    print("=== V16.01.4 REAL SNAPSHOT ADAPTER TEST ===")
    print(adapt_snapshot_source(demo))
