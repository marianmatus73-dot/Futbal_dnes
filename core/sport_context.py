from __future__ import annotations

import sqlite3
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sport_provider_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL, sport TEXT NOT NULL,
                    external_event_id TEXT NOT NULL, event TEXT NOT NULL,
                    start_time TEXT, captured_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL UNIQUE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_provider_snapshot_event "
                "ON sport_provider_snapshots(provider, sport, external_event_id, captured_at)"
            )

    def store_provider_snapshot(
        self,
        *,
        provider: str,
        sport: str,
        external_event_id: str,
        event: str,
        start_time: str,
        payload: dict[str, Any],
        captured_at: str | None = None,
    ) -> bool:
        """Persist an immutable provider response without storing credentials."""
        self.init_db()
        captured = captured_at or datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(
            f"{provider}|{sport}|{external_event_id}|{payload_json}".encode("utf-8")
        ).hexdigest()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO sport_provider_snapshots (
                    provider, sport, external_event_id, event, start_time,
                    captured_at, payload_json, payload_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider, sport, external_event_id, event, start_time,
                    captured, payload_json, payload_hash,
                ),
            )
        return cursor.rowcount == 1

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


