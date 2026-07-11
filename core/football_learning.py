from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.football_trainer import (
    DEFAULT_METADATA_PATH,
    MIN_TRAINING_SAMPLES,
    ensure_feature_history_table,
    train,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_result(value: Any) -> str:
    result = str(value or "").strip().upper()

    if result == "V":
        return "WON"

    if result == "P":
        return "LOST"

    if result in {"WON", "LOST", "VOID", "OPEN"}:
        return result

    return result


@dataclass
class FootballLearningResult:
    synced_features: int
    settled_features: int
    open_features: int
    trained: bool
    training_skipped_reason: str


class FootballLearningManager:
    def __init__(
        self,
        settings: Settings,
        *,
        metadata_path: str = DEFAULT_METADATA_PATH,
        min_samples: int = MIN_TRAINING_SAMPLES,
    ) -> None:
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")
        self.metadata_path = Path(metadata_path)
        self.min_samples = max(20, int(min_samples))

        ensure_feature_history_table(self.db_file)

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

    @staticmethod
    def _columns(
        conn: sqlite3.Connection,
        table_name: str,
    ) -> set[str]:
        return {
            str(row[1])
            for row in conn.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
        }

    def sync_feature_results(self) -> int:
        """
        Copies settled football results from sport_bets to
        football_feature_history.

        Primary join:
            source_hash

        Fallback join:
            league + event + selection + start/commence time
        """
        with self.connect() as conn:
            if not self._table_exists(conn, "sport_bets"):
                return 0

            bet_columns = self._columns(conn, "sport_bets")

            if "result" not in bet_columns:
                return 0

            has_settled_at = "settled_at" in bet_columns
            settled_expression = (
                "COALESCE(NULLIF(TRIM(b.settled_at), ''), ?)"
                if has_settled_at
                else "?"
            )

            before = conn.total_changes

            if "source_hash" in bet_columns:
                conn.execute(
                    f"""
                    UPDATE football_feature_history AS f
                    SET
                        result = (
                            SELECT CASE
                                WHEN UPPER(TRIM(b.result))='V' THEN 'WON'
                                WHEN UPPER(TRIM(b.result))='P' THEN 'LOST'
                                ELSE UPPER(TRIM(b.result))
                            END
                            FROM sport_bets AS b
                            WHERE b.source_hash=f.source_hash
                              AND b.sport='football'
                              AND UPPER(TRIM(b.result))
                                  IN ('WON','LOST','VOID','V','P')
                            ORDER BY b.id DESC
                            LIMIT 1
                        ),
                        settled_at = (
                            SELECT {settled_expression}
                            FROM sport_bets AS b
                            WHERE b.source_hash=f.source_hash
                              AND b.sport='football'
                              AND UPPER(TRIM(b.result))
                                  IN ('WON','LOST','VOID','V','P')
                            ORDER BY b.id DESC
                            LIMIT 1
                        )
                    WHERE f.result='OPEN'
                      AND EXISTS (
                          SELECT 1
                          FROM sport_bets AS b
                          WHERE b.source_hash=f.source_hash
                            AND b.sport='football'
                            AND UPPER(TRIM(b.result))
                                IN ('WON','LOST','VOID','V','P')
                      )
                    """,
                    (now_utc(),),
                )

            # Fallback for historical rows where hashes differ.
            required = {
                "league",
                "event",
                "selection",
                "start_time",
            }

            if required.issubset(bet_columns):
                conn.execute(
                    f"""
                    UPDATE football_feature_history AS f
                    SET
                        result = (
                            SELECT CASE
                                WHEN UPPER(TRIM(b.result))='V' THEN 'WON'
                                WHEN UPPER(TRIM(b.result))='P' THEN 'LOST'
                                ELSE UPPER(TRIM(b.result))
                            END
                            FROM sport_bets AS b
                            WHERE b.sport='football'
                              AND b.league=f.league
                              AND b.event=f.event
                              AND b.selection=f.selection
                              AND b.start_time=f.commence_time
                              AND UPPER(TRIM(b.result))
                                  IN ('WON','LOST','VOID','V','P')
                            ORDER BY b.id DESC
                            LIMIT 1
                        ),
                        settled_at = (
                            SELECT {settled_expression}
                            FROM sport_bets AS b
                            WHERE b.sport='football'
                              AND b.league=f.league
                              AND b.event=f.event
                              AND b.selection=f.selection
                              AND b.start_time=f.commence_time
                              AND UPPER(TRIM(b.result))
                                  IN ('WON','LOST','VOID','V','P')
                            ORDER BY b.id DESC
                            LIMIT 1
                        )
                    WHERE f.result='OPEN'
                      AND EXISTS (
                          SELECT 1
                          FROM sport_bets AS b
                          WHERE b.sport='football'
                            AND b.league=f.league
                            AND b.event=f.event
                            AND b.selection=f.selection
                            AND b.start_time=f.commence_time
                            AND UPPER(TRIM(b.result))
                                IN ('WON','LOST','VOID','V','P')
                      )
                    """,
                    (now_utc(),),
                )

            synced = conn.total_changes - before
            conn.commit()

        return synced

    def counts(self) -> tuple[int, int]:
        with self.connect() as conn:
            settled = conn.execute(
                """
                SELECT COUNT(*)
                FROM football_feature_history
                WHERE result IN ('WON','LOST')
                """
            ).fetchone()[0]

            open_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM football_feature_history
                WHERE result='OPEN'
                """
            ).fetchone()[0]

        return int(settled), int(open_count)

    def trained_sample_count(self) -> int:
        if not self.metadata_path.exists():
            return 0

        try:
            payload = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )

            return int(payload.get("samples", 0) or 0)
        except Exception:
            return 0

    def run(self) -> FootballLearningResult:
        synced = self.sync_feature_results()
        settled, open_count = self.counts()
        trained_samples = self.trained_sample_count()

        if settled < self.min_samples:
            return FootballLearningResult(
                synced_features=synced,
                settled_features=settled,
                open_features=open_count,
                trained=False,
                training_skipped_reason=(
                    f"not enough settled football features: "
                    f"{settled}/{self.min_samples}"
                ),
            )

        if settled <= trained_samples:
            return FootballLearningResult(
                synced_features=synced,
                settled_features=settled,
                open_features=open_count,
                trained=False,
                training_skipped_reason=(
                    "no new settled football samples since last training"
                ),
            )

        trained = train(
            self.db_file,
            min_samples=self.min_samples,
        )

        return FootballLearningResult(
            synced_features=synced,
            settled_features=settled,
            open_features=open_count,
            trained=trained,
            training_skipped_reason=(
                ""
                if trained
                else "trainer declined or failed to create a model"
            ),
        )


def run_football_learning(
    settings: Settings,
    *,
    min_samples: int = MIN_TRAINING_SAMPLES,
) -> FootballLearningResult:
    manager = FootballLearningManager(
        settings,
        min_samples=min_samples,
    )

    return manager.run()


if __name__ == "__main__":
    settings = Settings.from_env()
    result = run_football_learning(settings)

    print(
        "Football learning: "
        f"synced={result.synced_features}, "
        f"settled={result.settled_features}, "
        f"open={result.open_features}, "
        f"trained={result.trained}"
    )

    if result.training_skipped_reason:
        print(result.training_skipped_reason)
