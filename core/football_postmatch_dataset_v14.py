from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def make_hash(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


@dataclass
class FootballPostmatchDatasetSummary:
    discovered: int
    inserted: int
    updated: int
    missing_closing_line: int
    total_rows: int


class FootballPostmatchDatasetV14:
    """
    Builds one durable training row per settled football selection.

    It joins:
    - the pre-match feature row,
    - the latest available closing-window market snapshot,
    - the settled result,
    - CLV derived from opening vs closing implied probability.

    This module does not invent xG. Genuine xG can be joined later through
    football_xg_history_v14 using the same event identity.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")
        self.init_db()

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
        return conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name=?
            """,
            (table_name,),
        ).fetchone() is not None

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

    def init_db(self) -> None:
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_postmatch_dataset_v14 (
                    source_hash TEXT PRIMARY KEY,

                    sport_key TEXT NOT NULL DEFAULT '',
                    league TEXT NOT NULL DEFAULT '',
                    event TEXT NOT NULL DEFAULT '',
                    selection TEXT NOT NULL DEFAULT '',
                    bookmaker TEXT NOT NULL DEFAULT '',
                    commence_time TEXT NOT NULL DEFAULT '',

                    result TEXT NOT NULL,
                    target INTEGER NOT NULL,

                    opening_odds REAL,
                    opening_probability REAL,
                    model_probability REAL,
                    market_probability REAL,

                    closing_odds REAL,
                    closing_probability REAL,
                    closing_captured_at TEXT,
                    closing_bookmaker TEXT,

                    clv_probability REAL,
                    clv_odds_ratio REAL,

                    xg_home REAL,
                    xg_away REAL,
                    elo_difference REAL,
                    form_difference REAL,
                    model_dispersion REAL,
                    raw_edge REAL,
                    market_overround REAL,
                    bookmaker_count INTEGER,

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_postmatch_dataset_v14
                ON football_postmatch_dataset_v14 (
                    league,
                    commence_time,
                    result
                )
                """
            )

            conn.commit()

    @staticmethod
    def _pick(
        row: sqlite3.Row,
        columns: set[str],
        *names: str,
        default: Any = None,
    ) -> Any:
        for name in names:
            if name in columns:
                return row[name]

        return default

    def _latest_closing_snapshot(
        self,
        conn: sqlite3.Connection,
        *,
        sport_key: str,
        event: str,
        selection: str,
        commence_time: str,
    ) -> sqlite3.Row | None:
        if not self._table_exists(
            conn,
            "football_market_snapshots_v14",
        ):
            return None

        return conn.execute(
            """
            SELECT *
            FROM football_market_snapshots_v14
            WHERE sport_key=?
              AND event=?
              AND selection=?
              AND commence_time=?
            ORDER BY
                is_closing_window DESC,
                captured_at DESC,
                selected_odds DESC
            LIMIT 1
            """,
            (
                sport_key,
                event,
                selection,
                commence_time,
            ),
        ).fetchone()

    @staticmethod
    def _closing_probability(
        snapshot: sqlite3.Row | None,
    ) -> float | None:
        if snapshot is None:
            return None

        value = safe_float(
            snapshot["market_selection_probability"],
            None,
        )

        if value is not None and value > 0:
            return value

        odds = safe_float(snapshot["selected_odds"], None)

        if odds is None or odds <= 1.0:
            return None

        return 1.0 / odds

    def rebuild(self) -> FootballPostmatchDatasetSummary:
        with self.connect() as conn:
            if not self._table_exists(
                conn,
                "football_feature_history",
            ):
                return FootballPostmatchDatasetSummary(
                    discovered=0,
                    inserted=0,
                    updated=0,
                    missing_closing_line=0,
                    total_rows=0,
                )

            columns = self._columns(
                conn,
                "football_feature_history",
            )

            rows = conn.execute(
                """
                SELECT *
                FROM football_feature_history
                WHERE UPPER(COALESCE(result, 'OPEN')) IN ('WON', 'LOST')
                """
            ).fetchall()

            inserted = 0
            updated = 0
            missing_closing = 0

            for row in rows:
                sport_key = normalize_text(
                    self._pick(
                        row,
                        columns,
                        "sport_key",
                        default="",
                    )
                )
                league = normalize_text(
                    self._pick(
                        row,
                        columns,
                        "league",
                        default="",
                    )
                )
                event = normalize_text(
                    self._pick(
                        row,
                        columns,
                        "event",
                        default="",
                    )
                )
                selection = normalize_text(
                    self._pick(
                        row,
                        columns,
                        "selection",
                        default="",
                    )
                )
                bookmaker = normalize_text(
                    self._pick(
                        row,
                        columns,
                        "bookmaker",
                        default="",
                    )
                )
                commence_time = normalize_text(
                    self._pick(
                        row,
                        columns,
                        "commence_time",
                        default="",
                    )
                )
                result = normalize_text(
                    self._pick(
                        row,
                        columns,
                        "result",
                        default="",
                    )
                ).upper()

                source_hash = normalize_text(
                    self._pick(
                        row,
                        columns,
                        "source_hash",
                        default="",
                    )
                )

                if not source_hash:
                    source_hash = make_hash(
                        sport_key,
                        league,
                        event,
                        selection,
                        commence_time,
                    )

                opening_odds = safe_float(
                    self._pick(
                        row,
                        columns,
                        "odds",
                        "opening_odds",
                    ),
                    None,
                )
                opening_probability = (
                    1.0 / opening_odds
                    if opening_odds is not None
                    and opening_odds > 1.0
                    else None
                )

                model_probability = safe_float(
                    self._pick(
                        row,
                        columns,
                        "final_probability",
                        "model_probability",
                        "model_consensus_probability",
                    ),
                    None,
                )
                market_probability = safe_float(
                    self._pick(
                        row,
                        columns,
                        "market_selection_probability",
                    ),
                    None,
                )

                closing = self._latest_closing_snapshot(
                    conn,
                    sport_key=sport_key,
                    event=event,
                    selection=selection,
                    commence_time=commence_time,
                )

                closing_probability = (
                    self._closing_probability(closing)
                )
                closing_odds = (
                    safe_float(
                        closing["selected_odds"],
                        None,
                    )
                    if closing is not None
                    else None
                )

                if closing is None:
                    missing_closing += 1

                clv_probability = (
                    closing_probability - opening_probability
                    if closing_probability is not None
                    and opening_probability is not None
                    else None
                )

                clv_odds_ratio = (
                    opening_odds / closing_odds - 1.0
                    if opening_odds is not None
                    and closing_odds is not None
                    and closing_odds > 0
                    else None
                )

                target = 1 if result == "WON" else 0
                timestamp = now_utc()

                existed = conn.execute(
                    """
                    SELECT 1
                    FROM football_postmatch_dataset_v14
                    WHERE source_hash=?
                    """,
                    (source_hash,),
                ).fetchone() is not None

                conn.execute(
                    """
                    INSERT INTO football_postmatch_dataset_v14 (
                        source_hash,
                        sport_key,
                        league,
                        event,
                        selection,
                        bookmaker,
                        commence_time,
                        result,
                        target,
                        opening_odds,
                        opening_probability,
                        model_probability,
                        market_probability,
                        closing_odds,
                        closing_probability,
                        closing_captured_at,
                        closing_bookmaker,
                        clv_probability,
                        clv_odds_ratio,
                        xg_home,
                        xg_away,
                        elo_difference,
                        form_difference,
                        model_dispersion,
                        raw_edge,
                        market_overround,
                        bookmaker_count,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(source_hash) DO UPDATE SET
                        result=excluded.result,
                        target=excluded.target,
                        closing_odds=excluded.closing_odds,
                        closing_probability=excluded.closing_probability,
                        closing_captured_at=excluded.closing_captured_at,
                        closing_bookmaker=excluded.closing_bookmaker,
                        clv_probability=excluded.clv_probability,
                        clv_odds_ratio=excluded.clv_odds_ratio,
                        updated_at=excluded.updated_at
                    """,
                    (
                        source_hash,
                        sport_key,
                        league,
                        event,
                        selection,
                        bookmaker,
                        commence_time,
                        result,
                        target,
                        opening_odds,
                        opening_probability,
                        model_probability,
                        market_probability,
                        closing_odds,
                        closing_probability,
                        (
                            normalize_text(
                                closing["captured_at"]
                            )
                            if closing is not None
                            else ""
                        ),
                        (
                            normalize_text(
                                closing["bookmaker"]
                            )
                            if closing is not None
                            else ""
                        ),
                        clv_probability,
                        clv_odds_ratio,
                        safe_float(
                            self._pick(
                                row,
                                columns,
                                "xg_home",
                            ),
                            None,
                        ),
                        safe_float(
                            self._pick(
                                row,
                                columns,
                                "xg_away",
                            ),
                            None,
                        ),
                        safe_float(
                            self._pick(
                                row,
                                columns,
                                "elo_difference",
                                "elo_diff",
                            ),
                            None,
                        ),
                        safe_float(
                            self._pick(
                                row,
                                columns,
                                "form_difference",
                                "form_diff",
                            ),
                            None,
                        ),
                        safe_float(
                            self._pick(
                                row,
                                columns,
                                "model_dispersion",
                                "dispersion",
                            ),
                            None,
                        ),
                        safe_float(
                            self._pick(
                                row,
                                columns,
                                "raw_edge",
                            ),
                            None,
                        ),
                        safe_float(
                            self._pick(
                                row,
                                columns,
                                "market_overround",
                            ),
                            None,
                        ),
                        self._pick(
                            row,
                            columns,
                            "bookmaker_count",
                            default=None,
                        ),
                        timestamp,
                        timestamp,
                    ),
                )

                if existed:
                    updated += 1
                else:
                    inserted += 1

            conn.commit()

            total_rows = conn.execute(
                """
                SELECT COUNT(*)
                FROM football_postmatch_dataset_v14
                """
            ).fetchone()[0]

        return FootballPostmatchDatasetSummary(
            discovered=len(rows),
            inserted=inserted,
            updated=updated,
            missing_closing_line=missing_closing,
            total_rows=int(total_rows),
        )


def rebuild_football_postmatch_dataset_v14(
    settings: Settings,
) -> FootballPostmatchDatasetSummary:
    return FootballPostmatchDatasetV14(
        settings
    ).rebuild()


if __name__ == "__main__":
    settings = Settings.from_env()
    summary = rebuild_football_postmatch_dataset_v14(
        settings
    )

    print(
        "Football Postmatch Dataset v14: "
        f"discovered={summary.discovered}, "
        f"inserted={summary.inserted}, "
        f"updated={summary.updated}, "
        f"missing_closing={summary.missing_closing_line}, "
        f"total={summary.total_rows}"
    )
