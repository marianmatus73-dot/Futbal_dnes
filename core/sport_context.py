from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from core.config import Settings


@dataclass
class SportContext:
    lineup_confirmed: bool = False
    injury_impact: float = 0.0
    suspension_impact: float = 0.0
    rest_days: float | None = None
    travel_km: float | None = None
    starting_pitcher_confirmed: bool = False
    starting_pitcher_edge: float = 0.0
    source: str = ""
    captured_at: str = ""

    @property
    def verified(self) -> bool:
        return bool(self.source and self.captured_at)


class SportContextDatabase:
    def __init__(self, settings: Settings):
        self.db_file = Path(settings.db_file or "bets.db")

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sport_context_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sport TEXT NOT NULL, league TEXT NOT NULL DEFAULT '',
                    event TEXT NOT NULL, external_event_id TEXT,
                    start_time TEXT,
                    lineup_confirmed INTEGER NOT NULL DEFAULT 0,
                    injury_impact REAL NOT NULL DEFAULT 0,
                    suspension_impact REAL NOT NULL DEFAULT 0,
                    rest_days REAL, travel_km REAL,
                    starting_pitcher_confirmed INTEGER NOT NULL DEFAULT 0,
                    starting_pitcher_edge REAL NOT NULL DEFAULT 0,
                    source TEXT NOT NULL, captured_at TEXT NOT NULL,
                    source_hash TEXT UNIQUE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_sport_context_event "
                "ON sport_context_features(sport, external_event_id, event, captured_at)"
            )

    def latest(self, sport: str, event: str, external_event_id: str = "") -> SportContext:
        self.init_db()
        identity = "external_event_id=?" if external_event_id else "event=?"
        value = external_event_id or event
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM sport_context_features
                WHERE sport=? AND {identity}
                  AND TRIM(source) <> '' AND TRIM(captured_at) <> ''
                ORDER BY captured_at DESC, id DESC LIMIT 1
                """,
                (sport, value),
            ).fetchone()
        if row is None:
            return SportContext()
        return SportContext(
            lineup_confirmed=bool(row["lineup_confirmed"]),
            injury_impact=max(0.0, min(.25, float(row["injury_impact"] or 0))),
            suspension_impact=max(0.0, min(.25, float(row["suspension_impact"] or 0))),
            rest_days=float(row["rest_days"]) if row["rest_days"] is not None else None,
            travel_km=float(row["travel_km"]) if row["travel_km"] is not None else None,
            starting_pitcher_confirmed=bool(row["starting_pitcher_confirmed"]),
            starting_pitcher_edge=max(-.15, min(.15, float(row["starting_pitcher_edge"] or 0))),
            source=str(row["source"]), captured_at=str(row["captured_at"]),
        )

