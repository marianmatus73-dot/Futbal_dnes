from __future__ import annotations

import sqlite3
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.football_team_aliases import team_similarity, teams_match


@dataclass
class SportContext:
    lineup_confirmed: bool = False
    injury_impact: float = 0.0
    suspension_impact: float = 0.0
    rest_days: float | None = None
    travel_km: float | None = None
    starting_pitcher_confirmed: bool = False
    starting_pitcher_edge: float = 0.0
    home_team: str = ""
    away_team: str = ""
    home_absence_impact: float = 0.0
    away_absence_impact: float = 0.0
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
            existing = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(sport_context_features)")
            }
            for column, definition in {
                "home_team": "TEXT NOT NULL DEFAULT ''",
                "away_team": "TEXT NOT NULL DEFAULT ''",
                "home_absence_impact": "REAL NOT NULL DEFAULT 0",
                "away_absence_impact": "REAL NOT NULL DEFAULT 0",
            }.items():
                if column not in existing:
                    conn.execute(
                        f"ALTER TABLE sport_context_features ADD COLUMN {column} {definition}"
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sport_event_identity_links (
                    sport TEXT NOT NULL, consumer_event_id TEXT NOT NULL,
                    provider TEXT NOT NULL, provider_event_id TEXT NOT NULL,
                    consumer_event TEXT NOT NULL, provider_event TEXT NOT NULL,
                    similarity REAL NOT NULL, linked_at TEXT NOT NULL,
                    PRIMARY KEY (sport, consumer_event_id, provider)
                )
                """
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

    @staticmethod
    def _split_event(event: str) -> tuple[str, str]:
        for separator in (" vs ", " v ", " - "):
            if separator in event:
                return tuple(part.strip() for part in event.split(separator, 1))  # type: ignore[return-value]
        return "", ""

    @staticmethod
    def _parse_time(value: str) -> datetime | None:
        raw = str(value or "").strip().replace("Z", "+00:00")
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def latest(
        self,
        sport: str,
        event: str,
        external_event_id: str = "",
        start_time: str = "",
    ) -> SportContext:
        self.init_db()
        with self.connect() as conn:
            row = None
            if external_event_id:
                link = conn.execute(
                    """
                    SELECT provider_event_id FROM sport_event_identity_links
                    WHERE sport=? AND consumer_event_id=? AND provider='sportmonks-v3'
                    """,
                    (sport, external_event_id),
                ).fetchone()
                identity = str(link[0]) if link else external_event_id
                row = conn.execute(
                    """
                    SELECT * FROM sport_context_features
                    WHERE sport=? AND external_event_id=?
                      AND TRIM(source) <> '' AND TRIM(captured_at) <> ''
                    ORDER BY captured_at DESC, id DESC LIMIT 1
                    """,
                    (sport, identity),
                ).fetchone()
            if row is None and not external_event_id:
                row = conn.execute(
                    """
                    SELECT * FROM sport_context_features
                    WHERE sport=? AND event=?
                      AND TRIM(source) <> '' AND TRIM(captured_at) <> ''
                    ORDER BY captured_at DESC, id DESC LIMIT 1
                    """,
                    (sport, event),
                ).fetchone()
            if row is None and sport == "football":
                home, away = self._split_event(event)
                target_time = self._parse_time(start_time)
                best = None
                best_score = 0.0
                rows = conn.execute(
                    """
                    SELECT * FROM sport_context_features
                    WHERE sport='football' AND TRIM(source) <> ''
                    ORDER BY captured_at DESC, id DESC LIMIT 200
                    """
                ).fetchall()
                for candidate in rows:
                    candidate_time = self._parse_time(candidate["start_time"])
                    if target_time and candidate_time:
                        tolerance = float(os.getenv("SPORT_CONTEXT_MATCH_HOURS", "3"))
                        if abs((target_time - candidate_time).total_seconds()) > tolerance * 3600:
                            continue
                    candidate_home = str(candidate["home_team"] or "")
                    candidate_away = str(candidate["away_team"] or "")
                    if not candidate_home or not candidate_away:
                        candidate_home, candidate_away = self._split_event(str(candidate["event"]))
                    score = min(
                        team_similarity(home, candidate_home),
                        team_similarity(away, candidate_away),
                    )
                    if score >= .84 and score > best_score:
                        best, best_score = candidate, score
                row = best
                if row is not None and external_event_id:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO sport_event_identity_links (
                            sport, consumer_event_id, provider, provider_event_id,
                            consumer_event, provider_event, similarity, linked_at
                        ) VALUES (?, ?, 'sportmonks-v3', ?, ?, ?, ?, ?)
                        """,
                        (
                            sport, external_event_id, str(row["external_event_id"]),
                            event, str(row["event"]), best_score,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
        if row is None:
            return SportContext()
        captured = self._parse_time(str(row["captured_at"]))
        max_age = float(os.getenv("SPORT_CONTEXT_MAX_AGE_HOURS", "24"))
        if captured and captured < datetime.now(timezone.utc) - timedelta(hours=max_age):
            return SportContext()
        return SportContext(
            lineup_confirmed=bool(row["lineup_confirmed"]),
            injury_impact=max(0.0, min(.25, float(row["injury_impact"] or 0))),
            suspension_impact=max(0.0, min(.25, float(row["suspension_impact"] or 0))),
            rest_days=float(row["rest_days"]) if row["rest_days"] is not None else None,
            travel_km=float(row["travel_km"]) if row["travel_km"] is not None else None,
            starting_pitcher_confirmed=bool(row["starting_pitcher_confirmed"]),
            starting_pitcher_edge=max(-.15, min(.15, float(row["starting_pitcher_edge"] or 0))),
            home_team=str(row["home_team"] or ""),
            away_team=str(row["away_team"] or ""),
            home_absence_impact=max(0.0, min(.05, float(row["home_absence_impact"] or 0))),
            away_absence_impact=max(0.0, min(.05, float(row["away_absence_impact"] or 0))),
            source=str(row["source"]), captured_at=str(row["captured_at"]),
        )

    def selection_availability_adjustment(self, context: SportContext, selection: str) -> float:
        if context.home_team and teams_match(selection, context.home_team):
            return context.away_absence_impact - context.home_absence_impact
        if context.away_team and teams_match(selection, context.away_team):
            return context.home_absence_impact - context.away_absence_impact
        return 0.0



