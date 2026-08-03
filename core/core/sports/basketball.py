"""
BASKETBALL adapter.

Connect sport-specific APIs and feature extraction here.
"""

from core.sports.registry import SPORT_PROFILES

PROFILE = SPORT_PROFILES["basketball"]


def normalize_event(raw: dict) -> dict:
    required = ("event_key", "home_name", "away_name")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return {
        "sport": PROFILE.name,
        "competition": raw.get("competition"),
        "event_key": str(raw["event_key"]),
        "event_time": raw.get("event_time"),
        "home_name": raw["home_name"],
        "away_name": raw["away_name"],
        "status": raw.get("status", "OPEN"),
        "home_score": raw.get("home_score"),
        "away_score": raw.get("away_score"),
        "result": raw.get("result"),
    }
