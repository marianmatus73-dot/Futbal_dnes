
"""
V16.07 REAL ODDS SOURCE CONNECTOR

Normalizes external odds data into V16 format.
"""

from datetime import datetime


def normalize_odds(source_row):
    return {
        "event_id": source_row["event_id"],
        "home_team": source_row["home_team"],
        "away_team": source_row["away_team"],
        "opening_odds": source_row.get("opening_odds"),
        "current_odds": source_row.get("current_odds"),
        "bookmaker": source_row.get("bookmaker", "unknown"),
        "timestamp": datetime.utcnow().isoformat(),
        "source_version": "V16.07"
    }


def connect(source_rows):
    rows = [normalize_odds(r) for r in source_rows]

    return {
        "version": "V16.07",
        "source_rows": len(source_rows),
        "normalized_rows": len(rows),
        "status": "READY",
        "rows": rows
    }
