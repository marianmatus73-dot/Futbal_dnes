from __future__ import annotations

from core.multisport_learning.database import connect
from core.sports.registry import SPORT_PROFILES


if __name__ == "__main__":
    with connect("multisport_learning.db") as conn:
        for index, sport in enumerate(SPORT_PROFILES, start=1):
            event_key = f"{sport}-demo-{index}"
            conn.execute(
                """
                INSERT OR IGNORE INTO ml_events (
                    sport, competition, event_key, home_name, away_name,
                    status, result
                )
                VALUES (?, ?, ?, ?, ?, 'SETTLED', 'HOME_WIN')
                """,
                (
                    sport,
                    "demo",
                    event_key,
                    f"{sport}_home",
                    f"{sport}_away",
                ),
            )
            conn.execute(
                """
                INSERT INTO ml_predictions (
                    sport, event_key, selection, model_probability,
                    market_probability, odds, confidence
                )
                VALUES (?, ?, 'HOME', 0.60, 0.52, 1.90, 0.75)
                """,
                (sport, event_key),
            )
            conn.execute(
                """
                INSERT INTO ml_market_snapshots (
                    sport, event_key, selection, odds, bookmaker, snapshot_type
                )
                VALUES (?, ?, 'HOME', 1.90, 'demo', 'OPEN')
                """,
                (sport, event_key),
            )
        conn.commit()
    print("Demo data inserted.")
