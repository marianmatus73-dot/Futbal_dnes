from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()

    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def split_event(event: str) -> tuple[str, str] | None:
    text = normalize_text(event)

    for separator in (" vs ", " v ", " - "):
        if separator in text:
            home, away = text.split(separator, 1)
            home = normalize_text(home)
            away = normalize_text(away)

            if home and away:
                return home, away

    return None


def make_hash(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


@dataclass
class FootballDataCollectorSummary:
    market_snapshots_added: int
    xg_rows_added: int
    market_snapshots_total: int
    xg_rows_total: int


class FootballDataCollectorV14:
    """
    Collects persistent pre-match market snapshots and mirrors genuine
    post-match xG rows for V14 training.

    Market probabilities are stored directly. Fair odds are derived from
    those probabilities and must not be confused with bookmaker closing
    prices unless captured close to kickoff.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        closing_window_hours: float = 12.0,
    ) -> None:
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")
        self.closing_window_hours = max(
            1.0,
            min(72.0, float(closing_window_hours)),
        )
        self.init_tables()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _table_exists(
        conn: sqlite3.Connection,
        table_name: str,
    ) -> bool:
        row = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name=?
            """,
            (table_name,),
        ).fetchone()

        return row is not None

    def init_tables(self) -> None:
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_market_snapshots_v14 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    snapshot_hash TEXT NOT NULL UNIQUE,
                    source_hash TEXT,
                    sport_key TEXT NOT NULL DEFAULT '',
                    league TEXT NOT NULL DEFAULT '',
                    event TEXT NOT NULL DEFAULT '',
                    home_team TEXT NOT NULL DEFAULT '',
                    away_team TEXT NOT NULL DEFAULT '',
                    selection TEXT NOT NULL DEFAULT '',
                    bookmaker TEXT NOT NULL DEFAULT '',

                    commence_time TEXT NOT NULL DEFAULT '',
                    captured_at TEXT NOT NULL,
                    hours_to_start REAL,

                    selected_odds REAL NOT NULL DEFAULT 0.0,

                    market_home_probability REAL NOT NULL DEFAULT 0.0,
                    market_draw_probability REAL NOT NULL DEFAULT 0.0,
                    market_away_probability REAL NOT NULL DEFAULT 0.0,
                    market_selection_probability REAL NOT NULL DEFAULT 0.0,

                    fair_home_odds REAL,
                    fair_draw_odds REAL,
                    fair_away_odds REAL,

                    market_overround REAL NOT NULL DEFAULT 0.0,
                    bookmaker_count INTEGER NOT NULL DEFAULT 0,

                    is_closing_window INTEGER NOT NULL DEFAULT 0
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_market_snapshots_event
                ON football_market_snapshots_v14 (
                    sport_key,
                    commence_time,
                    event,
                    captured_at
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_xg_history_v14 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    source_hash TEXT NOT NULL UNIQUE,
                    league TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,

                    home_xg REAL NOT NULL,
                    away_xg REAL NOT NULL,
                    home_goals INTEGER,
                    away_goals INTEGER,

                    played_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    collected_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_xg_history_v14_league
                ON football_xg_history_v14 (
                    league,
                    played_at
                )
                """
            )

            conn.commit()

    @staticmethod
    def _fair_odds(probability: float) -> float | None:
        probability = safe_float(probability)

        if probability <= 0.0:
            return None

        return 1.0 / probability

    def collect_market_snapshots(self) -> int:
        captured_at = datetime.now(timezone.utc)

        with self.connect() as conn:
            if not self._table_exists(
                conn,
                "football_feature_history",
            ):
                return 0

            rows = conn.execute(
                """
                SELECT
                    source_hash,
                    sport_key,
                    league,
                    event,
                    selection,
                    bookmaker,
                    commence_time,
                    odds,
                    market_home_probability,
                    market_draw_probability,
                    market_away_probability,
                    market_selection_probability,
                    market_overround,
                    bookmaker_count,
                    result
                FROM football_feature_history
                WHERE result='OPEN'
                """
            ).fetchall()

            added = 0

            for row in rows:
                teams = split_event(row["event"])

                if teams is None:
                    continue

                commence = parse_datetime(row["commence_time"])

                if commence is None:
                    continue

                hours_to_start = (
                    commence - captured_at
                ).total_seconds() / 3600.0

                # Do not store stale snapshots long after kickoff.
                if hours_to_start < -3.0:
                    continue

                is_closing_window = int(
                    0.0 <= hours_to_start <= self.closing_window_hours
                )

                # Bucket snapshots hourly to prevent repeated duplicates.
                bucket = captured_at.replace(
                    minute=0,
                    second=0,
                    microsecond=0,
                ).isoformat()

                source_hash = normalize_text(row["source_hash"])
                snapshot_hash = make_hash(
                    source_hash,
                    row["sport_key"],
                    row["event"],
                    row["selection"],
                    row["bookmaker"],
                    bucket,
                )

                home_probability = safe_float(
                    row["market_home_probability"]
                )
                draw_probability = safe_float(
                    row["market_draw_probability"]
                )
                away_probability = safe_float(
                    row["market_away_probability"]
                )

                before = conn.total_changes

                conn.execute(
                    """
                    INSERT OR IGNORE INTO football_market_snapshots_v14 (
                        snapshot_hash,
                        source_hash,
                        sport_key,
                        league,
                        event,
                        home_team,
                        away_team,
                        selection,
                        bookmaker,
                        commence_time,
                        captured_at,
                        hours_to_start,
                        selected_odds,
                        market_home_probability,
                        market_draw_probability,
                        market_away_probability,
                        market_selection_probability,
                        fair_home_odds,
                        fair_draw_odds,
                        fair_away_odds,
                        market_overround,
                        bookmaker_count,
                        is_closing_window
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        snapshot_hash,
                        source_hash,
                        normalize_text(row["sport_key"]),
                        normalize_text(row["league"]),
                        normalize_text(row["event"]),
                        teams[0],
                        teams[1],
                        normalize_text(row["selection"]),
                        normalize_text(row["bookmaker"]),
                        normalize_text(row["commence_time"]),
                        captured_at.isoformat(timespec="seconds"),
                        hours_to_start,
                        safe_float(row["odds"]),
                        home_probability,
                        draw_probability,
                        away_probability,
                        safe_float(
                            row["market_selection_probability"]
                        ),
                        self._fair_odds(home_probability),
                        self._fair_odds(draw_probability),
                        self._fair_odds(away_probability),
                        safe_float(row["market_overround"]),
                        int(row["bookmaker_count"] or 0),
                        is_closing_window,
                    ),
                )

                if conn.total_changes > before:
                    added += 1

            conn.commit()

        return added

    def sync_real_xg_history(self) -> int:
        """
        Mirror only genuine xG records already stored by result learning.
        No goal-as-xG proxy is created here.
        """
        with self.connect() as conn:
            if not self._table_exists(conn, "football_xg_history"):
                return 0

            rows = conn.execute(
                """
                SELECT *
                FROM football_xg_history
                ORDER BY id
                """
            ).fetchall()

            added = 0

            for row in rows:
                source_hash = normalize_text(row["source_hash"])

                if not source_hash:
                    source_hash = make_hash(
                        row["league"],
                        row["home_team"],
                        row["away_team"],
                        row["played_at"],
                        row["home_xg"],
                        row["away_xg"],
                    )

                before = conn.total_changes

                conn.execute(
                    """
                    INSERT OR IGNORE INTO football_xg_history_v14 (
                        source_hash,
                        league,
                        home_team,
                        away_team,
                        home_xg,
                        away_xg,
                        home_goals,
                        away_goals,
                        played_at,
                        source,
                        collected_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_hash,
                        normalize_text(row["league"]),
                        normalize_text(row["home_team"]),
                        normalize_text(row["away_team"]),
                        safe_float(row["home_xg"]),
                        safe_float(row["away_xg"]),
                        row["home_goals"],
                        row["away_goals"],
                        normalize_text(row["played_at"]),
                        normalize_text(row["source"]),
                        now_utc(),
                    ),
                )

                if conn.total_changes > before:
                    added += 1

            conn.commit()

        return added

    def status(self) -> tuple[int, int]:
        with self.connect() as conn:
            market_total = conn.execute(
                """
                SELECT COUNT(*)
                FROM football_market_snapshots_v14
                """
            ).fetchone()[0]

            xg_total = conn.execute(
                """
                SELECT COUNT(*)
                FROM football_xg_history_v14
                """
            ).fetchone()[0]

        return int(market_total), int(xg_total)

    def run(self) -> FootballDataCollectorSummary:
        market_added = self.collect_market_snapshots()
        xg_added = self.sync_real_xg_history()
        market_total, xg_total = self.status()

        return FootballDataCollectorSummary(
            market_snapshots_added=market_added,
            xg_rows_added=xg_added,
            market_snapshots_total=market_total,
            xg_rows_total=xg_total,
        )


def run_football_data_collector_v14(
    settings: Settings,
    *,
    closing_window_hours: float = 12.0,
) -> FootballDataCollectorSummary:
    return FootballDataCollectorV14(
        settings,
        closing_window_hours=closing_window_hours,
    ).run()


if __name__ == "__main__":
    import os

    settings = Settings.from_env()
    summary = run_football_data_collector_v14(
        settings,
        closing_window_hours=float(
            os.getenv(
                "FOOTBALL_CLOSING_WINDOW_HOURS",
                "12",
            )
        ),
    )

    print(
        "Football Data Collector v14: "
        f"market_added={summary.market_snapshots_added}, "
        f"xg_added={summary.xg_rows_added}, "
        f"market_total={summary.market_snapshots_total}, "
        f"xg_total={summary.xg_rows_total}"
    )
