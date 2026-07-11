from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.football_features import FootballFeatures
from core.football_trainer import FEATURE_ORDER, ensure_feature_history_table


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_result(value: Any) -> str:
    result = str(value or "").strip().upper()

    if result in {"", "OPEN"}:
        return "OPEN"

    if result == "V":
        return "WON"

    if result == "P":
        return "LOST"

    if result in {"WON", "LOST", "VOID"}:
        return result

    return result


def make_feature_hash(features: FootballFeatures) -> str:
    raw = "|".join(
        [
            features.sport_key,
            features.league,
            features.event,
            features.selection,
            features.bookmaker,
            features.commence_time,
        ]
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class FootballFeatureStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")
        ensure_feature_history_table(self.db_file)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save_open_feature(
        self,
        features: FootballFeatures,
        *,
        source_hash: str | None = None,
    ) -> bool:
        feature_hash = source_hash or make_feature_hash(features)
        payload = features.to_dict()

        columns = [
            "created_at",
            "settled_at",
            "sport_key",
            "league",
            "event",
            "selection",
            "bookmaker",
            "commence_time",
            *FEATURE_ORDER,
            "result",
            "source_hash",
        ]

        values = [
            now_utc(),
            None,
            features.sport_key,
            features.league,
            features.event,
            features.selection,
            features.bookmaker,
            features.commence_time,
            *[
                float(payload.get(name, 0.0) or 0.0)
                for name in FEATURE_ORDER
            ],
            "OPEN",
            feature_hash,
        ]

        placeholders = ",".join("?" for _ in columns)
        column_sql = ",".join(columns)

        with self.connect() as conn:
            before = conn.total_changes

            conn.execute(
                f"""
                INSERT OR IGNORE INTO football_feature_history (
                    {column_sql}
                )
                VALUES ({placeholders})
                """,
                values,
            )

            inserted = conn.total_changes > before
            conn.commit()

        return inserted

    def settle_feature(
        self,
        *,
        source_hash: str,
        result: str,
        settled_at: str | None = None,
    ) -> bool:
        normalized = normalize_result(result)

        if normalized not in {"WON", "LOST", "VOID"}:
            raise ValueError(
                f"Unsupported settled result: {result}"
            )

        with self.connect() as conn:
            before = conn.total_changes

            conn.execute(
                """
                UPDATE football_feature_history
                SET result=?, settled_at=?
                WHERE source_hash=?
                  AND result='OPEN'
                """,
                (
                    normalized,
                    settled_at or now_utc(),
                    source_hash,
                ),
            )

            updated = conn.total_changes > before
            conn.commit()

        return updated

    def settle_by_bet_identity(
        self,
        *,
        sport_key: str,
        league: str,
        event: str,
        selection: str,
        commence_time: str,
        result: str,
        settled_at: str | None = None,
    ) -> int:
        normalized = normalize_result(result)

        if normalized not in {"WON", "LOST", "VOID"}:
            raise ValueError(
                f"Unsupported settled result: {result}"
            )

        with self.connect() as conn:
            before = conn.total_changes

            conn.execute(
                """
                UPDATE football_feature_history
                SET result=?, settled_at=?
                WHERE sport_key=?
                  AND league=?
                  AND event=?
                  AND selection=?
                  AND commence_time=?
                  AND result='OPEN'
                """,
                (
                    normalized,
                    settled_at or now_utc(),
                    sport_key,
                    league,
                    event,
                    selection,
                    commence_time,
                ),
            )

            updated = conn.total_changes - before
            conn.commit()

        return updated

    def count_by_result(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT result, COUNT(*) AS total
                FROM football_feature_history
                GROUP BY result
                ORDER BY total DESC
                """
            ).fetchall()

        return {
            str(row["result"]): int(row["total"])
            for row in rows
        }

    def delete_open_duplicates(self) -> int:
        """
        Keeps one OPEN feature row per unique bet identity.
        Settled rows are never removed.
        """
        with self.connect() as conn:
            before = conn.total_changes

            conn.execute(
                """
                DELETE FROM football_feature_history
                WHERE result='OPEN'
                  AND id NOT IN (
                      SELECT MIN(id)
                      FROM football_feature_history
                      WHERE result='OPEN'
                      GROUP BY
                          sport_key,
                          league,
                          event,
                          selection,
                          bookmaker,
                          commence_time
                  )
                """
            )

            deleted = conn.total_changes - before
            conn.commit()

        return deleted


def save_football_features(
    settings: Settings,
    features: FootballFeatures,
    *,
    source_hash: str | None = None,
) -> bool:
    store = FootballFeatureStore(settings)

    return store.save_open_feature(
        features,
        source_hash=source_hash,
    )


def settle_football_features(
    settings: Settings,
    *,
    source_hash: str,
    result: str,
    settled_at: str | None = None,
) -> bool:
    store = FootballFeatureStore(settings)

    return store.settle_feature(
        source_hash=source_hash,
        result=result,
        settled_at=settled_at,
    )
