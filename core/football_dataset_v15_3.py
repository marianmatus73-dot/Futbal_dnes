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


def safe_int(
    value: Any,
    default: int | None = None,
) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def make_hash(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def validate_insert_shape() -> tuple[int, int]:
    """
    Validate the main INSERT statement at source level.

    Returns:
        (column_count, placeholder_count)

    Raises:
        RuntimeError when the SQL shape is inconsistent.
    """
    columns = (
        "source_hash",
        "sport_key",
        "league",
        "event",
        "home_team",
        "away_team",
        "selection",
        "bookmaker",
        "commence_time",
        "result",
        "target",
        "opening_odds",
        "opening_probability",
        "model_probability",
        "market_probability",
        "consensus_probability",
        "closing_odds",
        "closing_probability",
        "closing_captured_at",
        "closing_bookmaker",
        "closing_is_window",
        "clv_probability",
        "clv_odds_ratio",
        "xg_home",
        "xg_away",
        "xg_difference",
        "elo_home",
        "elo_away",
        "elo_difference",
        "form_home",
        "form_away",
        "form_difference",
        "model_dispersion",
        "consensus_safety",
        "raw_edge",
        "market_overround",
        "bookmaker_count",
        "confidence",
        "risk",
        "has_closing_line",
        "has_xg",
        "has_elo",
        "has_form",
        "training_ready",
        "feature_version",
        "created_at",
        "updated_at",
    )

    placeholder_count = 47

    if len(columns) != placeholder_count:
        raise RuntimeError(
            "football_dataset_v15 INSERT shape mismatch: "
            f"columns={len(columns)}, placeholders={placeholder_count}"
        )

    return len(columns), placeholder_count


@dataclass
class FootballDatasetV15Summary:
    discovered: int
    inserted: int
    updated: int
    missing_model_probability: int
    missing_market_probability: int
    missing_opening_odds: int
    with_closing: int
    with_xg: int
    with_elo: int
    with_form: int
    training_ready: int
    total_rows: int


class FootballDatasetV15:
    """
    Unified football training dataset.

    One row represents one settled football selection and combines:
    - pre-match model features,
    - market and opening price,
    - closing-line information,
    - genuine post-match xG when available,
    - team ELO/form context,
    - target label and data-quality flags.

    No synthetic xG or fake closing price is generated.
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

    def init_db(self) -> None:
        validate_insert_shape()
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_dataset_v15 (
                    source_hash TEXT PRIMARY KEY,

                    sport_key TEXT NOT NULL DEFAULT '',
                    league TEXT NOT NULL DEFAULT '',
                    event TEXT NOT NULL DEFAULT '',
                    home_team TEXT NOT NULL DEFAULT '',
                    away_team TEXT NOT NULL DEFAULT '',
                    selection TEXT NOT NULL DEFAULT '',
                    bookmaker TEXT NOT NULL DEFAULT '',
                    commence_time TEXT NOT NULL DEFAULT '',

                    result TEXT NOT NULL,
                    target INTEGER NOT NULL,

                    opening_odds REAL,
                    opening_probability REAL,
                    model_probability REAL,
                    market_probability REAL,
                    consensus_probability REAL,

                    closing_odds REAL,
                    closing_probability REAL,
                    closing_captured_at TEXT,
                    closing_bookmaker TEXT,
                    closing_is_window INTEGER NOT NULL DEFAULT 0,

                    clv_probability REAL,
                    clv_odds_ratio REAL,

                    xg_home REAL,
                    xg_away REAL,
                    xg_difference REAL,

                    elo_home REAL,
                    elo_away REAL,
                    elo_difference REAL,

                    form_home REAL,
                    form_away REAL,
                    form_difference REAL,

                    model_dispersion REAL,
                    consensus_safety REAL,
                    raw_edge REAL,
                    market_overround REAL,
                    bookmaker_count INTEGER,
                    confidence REAL,
                    risk TEXT,

                    has_closing_line INTEGER NOT NULL DEFAULT 0,
                    has_xg INTEGER NOT NULL DEFAULT 0,
                    has_elo INTEGER NOT NULL DEFAULT 0,
                    has_form INTEGER NOT NULL DEFAULT 0,
                    training_ready INTEGER NOT NULL DEFAULT 0,

                    feature_version TEXT NOT NULL DEFAULT 'v15',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            columns = self._columns(
                conn,
                "football_dataset_v15",
            )

            if "consensus_safety" not in columns:
                conn.execute(
                    """
                    ALTER TABLE football_dataset_v15
                    ADD COLUMN consensus_safety REAL
                    """
                )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_dataset_v15_training
                ON football_dataset_v15 (
                    training_ready,
                    league,
                    commence_time
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_dataset_v15_result
                ON football_dataset_v15 (
                    result,
                    target
                )
                """
            )

            conn.commit()

    def _closing_snapshot(
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

    def _xg_row(
        self,
        conn: sqlite3.Connection,
        *,
        league: str,
        home_team: str,
        away_team: str,
        commence_time: str,
    ) -> sqlite3.Row | None:
        table_name = ""

        if self._table_exists(
            conn,
            "football_xg_history_v14",
        ):
            table_name = "football_xg_history_v14"
        elif self._table_exists(
            conn,
            "football_xg_history",
        ):
            table_name = "football_xg_history"

        if not table_name:
            return None

        columns = self._columns(conn, table_name)
        played_column = (
            "played_at"
            if "played_at" in columns
            else "created_at"
            if "created_at" in columns
            else ""
        )

        if not played_column:
            return None

        return conn.execute(
            f"""
            SELECT *
            FROM {table_name}
            WHERE league=?
              AND home_team=?
              AND away_team=?
            ORDER BY
                ABS(
                    julianday({played_column})
                    - julianday(?)
                ) ASC
            LIMIT 1
            """,
            (
                league,
                home_team,
                away_team,
                commence_time,
            ),
        ).fetchone()

    def _elo_context(
        self,
        conn: sqlite3.Connection,
        *,
        league: str,
        home_team: str,
        away_team: str,
    ) -> tuple[float | None, float | None]:
        table_name = ""

        if self._table_exists(
            conn,
            "football_team_elo_v14",
        ):
            table_name = "football_team_elo_v14"
        elif self._table_exists(
            conn,
            "football_elo_ratings",
        ):
            table_name = "football_elo_ratings"

        if not table_name:
            return None, None

        columns = self._columns(conn, table_name)
        rating_column = (
            "rating"
            if "rating" in columns
            else "elo"
            if "elo" in columns
            else ""
        )

        if not rating_column:
            return None, None

        home = conn.execute(
            f"""
            SELECT {rating_column}
            FROM {table_name}
            WHERE team=? AND league=?
            LIMIT 1
            """,
            (home_team, league),
        ).fetchone()

        away = conn.execute(
            f"""
            SELECT {rating_column}
            FROM {table_name}
            WHERE team=? AND league=?
            LIMIT 1
            """,
            (away_team, league),
        ).fetchone()

        return (
            safe_float(home[0], None)
            if home is not None
            else None,
            safe_float(away[0], None)
            if away is not None
            else None,
        )

    def _form_context(
        self,
        conn: sqlite3.Connection,
        *,
        league: str,
        home_team: str,
        away_team: str,
    ) -> tuple[float | None, float | None]:
        if not self._table_exists(
            conn,
            "football_team_form",
        ):
            return None, None

        columns = self._columns(
            conn,
            "football_team_form",
        )
        score_column = (
            "form_score"
            if "form_score" in columns
            else "recent_form"
            if "recent_form" in columns
            else "rating"
            if "rating" in columns
            else ""
        )

        if not score_column:
            return None, None

        home = conn.execute(
            f"""
            SELECT {score_column}
            FROM football_team_form
            WHERE team=? AND league=?
            LIMIT 1
            """,
            (home_team, league),
        ).fetchone()

        away = conn.execute(
            f"""
            SELECT {score_column}
            FROM football_team_form
            WHERE team=? AND league=?
            LIMIT 1
            """,
            (away_team, league),
        ).fetchone()

        return (
            safe_float(home[0], None)
            if home is not None
            else None,
            safe_float(away[0], None)
            if away is not None
            else None,
        )

    @staticmethod
    def _selection_probability(
        row: sqlite3.Row | None,
    ) -> float | None:
        if row is None:
            return None

        probability = safe_float(
            row["market_selection_probability"],
            None,
        )

        if probability is not None and probability > 0:
            return probability

        odds = safe_float(row["selected_odds"], None)

        if odds is None or odds <= 1.0:
            return None

        return 1.0 / odds

    @staticmethod
    def _split_event(
        event: str,
    ) -> tuple[str, str]:
        for separator in (" vs ", " v ", " - "):
            if separator in event:
                home, away = event.split(
                    separator,
                    1,
                )
                return (
                    normalize_text(home),
                    normalize_text(away),
                )

        return "", ""

    def rebuild(self) -> FootballDatasetV15Summary:
        with self.connect() as conn:
            if not self._table_exists(
                conn,
                "football_feature_history",
            ):
                return FootballDatasetV15Summary(
                    discovered=0,
                    inserted=0,
                    updated=0,
                    missing_model_probability=0,
                    missing_market_probability=0,
                    missing_opening_odds=0,
                    with_closing=0,
                    with_xg=0,
                    with_elo=0,
                    with_form=0,
                    training_ready=0,
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
                WHERE UPPER(COALESCE(result, 'OPEN'))
                    IN ('WON', 'LOST')
                """
            ).fetchall()

            inserted = 0
            updated = 0
            missing_model_probability = 0
            missing_market_probability = 0
            missing_opening_odds = 0
            with_closing = 0
            with_xg = 0
            with_elo = 0
            with_form = 0
            ready_count = 0

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

                home_team = normalize_text(
                    self._pick(
                        row,
                        columns,
                        "home_team",
                        default="",
                    )
                )
                away_team = normalize_text(
                    self._pick(
                        row,
                        columns,
                        "away_team",
                        default="",
                    )
                )

                if not home_team or not away_team:
                    home_team, away_team = (
                        self._split_event(event)
                    )

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
                        "consensus_probability",
                        "predicted_probability",
                    ),
                    None,
                )
                consensus_probability = safe_float(
                    self._pick(
                        row,
                        columns,
                        "model_consensus_probability",
                        "consensus_probability",
                    ),
                    None,
                )
                market_probability = safe_float(
                    self._pick(
                        row,
                        columns,
                        "market_selection_probability",
                        "market_probability",
                        "implied_probability",
                    ),
                    None,
                )

                if (
                    market_probability is None
                    and opening_probability is not None
                ):
                    market_probability = opening_probability

                closing = self._closing_snapshot(
                    conn,
                    sport_key=sport_key,
                    event=event,
                    selection=selection,
                    commence_time=commence_time,
                )

                closing_probability = (
                    self._selection_probability(closing)
                )
                closing_odds = (
                    safe_float(
                        closing["selected_odds"],
                        None,
                    )
                    if closing is not None
                    else None
                )
                closing_is_window = (
                    int(closing["is_closing_window"] or 0)
                    if closing is not None
                    else 0
                )
                has_closing = int(
                    closing_probability is not None
                    and closing_odds is not None
                )

                if has_closing:
                    with_closing += 1

                xg = self._xg_row(
                    conn,
                    league=league,
                    home_team=home_team,
                    away_team=away_team,
                    commence_time=commence_time,
                )

                xg_home = (
                    safe_float(xg["home_xg"], None)
                    if xg is not None
                    else None
                )
                xg_away = (
                    safe_float(xg["away_xg"], None)
                    if xg is not None
                    else None
                )
                has_xg = int(
                    xg_home is not None
                    and xg_away is not None
                )

                if has_xg:
                    with_xg += 1

                elo_home, elo_away = self._elo_context(
                    conn,
                    league=league,
                    home_team=home_team,
                    away_team=away_team,
                )
                has_elo = int(
                    elo_home is not None
                    and elo_away is not None
                )

                if has_elo:
                    with_elo += 1

                form_home, form_away = (
                    self._form_context(
                        conn,
                        league=league,
                        home_team=home_team,
                        away_team=away_team,
                    )
                )
                has_form = int(
                    form_home is not None
                    and form_away is not None
                )

                if has_form:
                    with_form += 1

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

                has_valid_target = result in {
                    "WON",
                    "LOST",
                }
                has_valid_model_probability = (
                    model_probability is not None
                    and 0.0 < model_probability < 1.0
                )
                has_valid_market_probability = (
                    market_probability is not None
                    and 0.0 < market_probability < 1.0
                )
                has_valid_opening_odds = (
                    opening_odds is not None
                    and opening_odds > 1.0
                )

                training_ready = int(
                    has_valid_target
                    and has_valid_model_probability
                    and has_valid_market_probability
                    and has_valid_opening_odds
                )

                if not has_valid_model_probability:
                    missing_model_probability += 1

                if not has_valid_market_probability:
                    missing_market_probability += 1

                if not has_valid_opening_odds:
                    missing_opening_odds += 1

                if training_ready:
                    ready_count += 1

                existed = conn.execute(
                    """
                    SELECT 1
                    FROM football_dataset_v15
                    WHERE source_hash=?
                    """,
                    (source_hash,),
                ).fetchone() is not None

                timestamp = now_utc()

                conn.execute(
                    """
                    INSERT INTO football_dataset_v15 (
                        source_hash,
                        sport_key,
                        league,
                        event,
                        home_team,
                        away_team,
                        selection,
                        bookmaker,
                        commence_time,
                        result,
                        target,
                        opening_odds,
                        opening_probability,
                        model_probability,
                        market_probability,
                        consensus_probability,
                        closing_odds,
                        closing_probability,
                        closing_captured_at,
                        closing_bookmaker,
                        closing_is_window,
                        clv_probability,
                        clv_odds_ratio,
                        xg_home,
                        xg_away,
                        xg_difference,
                        elo_home,
                        elo_away,
                        elo_difference,
                        form_home,
                        form_away,
                        form_difference,
                        model_dispersion,
                        consensus_safety,
                        raw_edge,
                        market_overround,
                        bookmaker_count,
                        confidence,
                        risk,
                        has_closing_line,
                        has_xg,
                        has_elo,
                        has_form,
                        training_ready,
                        feature_version,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(source_hash) DO UPDATE SET
                        result=excluded.result,
                        target=excluded.target,
                        model_probability=excluded.model_probability,
                        market_probability=excluded.market_probability,
                        consensus_probability=excluded.consensus_probability,
                        closing_odds=excluded.closing_odds,
                        closing_probability=excluded.closing_probability,
                        closing_captured_at=excluded.closing_captured_at,
                        closing_bookmaker=excluded.closing_bookmaker,
                        closing_is_window=excluded.closing_is_window,
                        clv_probability=excluded.clv_probability,
                        clv_odds_ratio=excluded.clv_odds_ratio,
                        xg_home=excluded.xg_home,
                        xg_away=excluded.xg_away,
                        xg_difference=excluded.xg_difference,
                        elo_home=excluded.elo_home,
                        elo_away=excluded.elo_away,
                        elo_difference=excluded.elo_difference,
                        form_home=excluded.form_home,
                        form_away=excluded.form_away,
                        form_difference=excluded.form_difference,
                        model_dispersion=excluded.model_dispersion,
                        consensus_safety=excluded.consensus_safety,
                        raw_edge=excluded.raw_edge,
                        market_overround=excluded.market_overround,
                        bookmaker_count=excluded.bookmaker_count,
                        confidence=excluded.confidence,
                        risk=excluded.risk,
                        has_closing_line=excluded.has_closing_line,
                        has_xg=excluded.has_xg,
                        has_elo=excluded.has_elo,
                        has_form=excluded.has_form,
                        training_ready=excluded.training_ready,
                        updated_at=excluded.updated_at
                    """,
                    (
                        source_hash,
                        sport_key,
                        league,
                        event,
                        home_team,
                        away_team,
                        selection,
                        bookmaker,
                        commence_time,
                        result,
                        1 if result == "WON" else 0,
                        opening_odds,
                        opening_probability,
                        model_probability,
                        market_probability,
                        consensus_probability,
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
                        closing_is_window,
                        clv_probability,
                        clv_odds_ratio,
                        xg_home,
                        xg_away,
                        (
                            xg_home - xg_away
                            if xg_home is not None
                            and xg_away is not None
                            else None
                        ),
                        elo_home,
                        elo_away,
                        (
                            elo_home - elo_away
                            if elo_home is not None
                            and elo_away is not None
                            else safe_float(
                                self._pick(
                                    row,
                                    columns,
                                    "elo_difference",
                                    "elo_diff",
                                ),
                                None,
                            )
                        ),
                        form_home,
                        form_away,
                        (
                            form_home - form_away
                            if form_home is not None
                            and form_away is not None
                            else safe_float(
                                self._pick(
                                    row,
                                    columns,
                                    "form_difference",
                                    "form_diff",
                                ),
                                None,
                            )
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
                                "consensus_safety",
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
                        safe_int(
                            self._pick(
                                row,
                                columns,
                                "bookmaker_count",
                            ),
                            None,
                        ),
                        safe_float(
                            self._pick(
                                row,
                                columns,
                                "confidence",
                            ),
                            None,
                        ),
                        normalize_text(
                            self._pick(
                                row,
                                columns,
                                "risk",
                                default="",
                            )
                        ),
                        has_closing,
                        has_xg,
                        has_elo,
                        has_form,
                        training_ready,
                        "v15",
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
                FROM football_dataset_v15
                """
            ).fetchone()[0]

        return FootballDatasetV15Summary(
            discovered=len(rows),
            inserted=inserted,
            updated=updated,
            missing_model_probability=missing_model_probability,
            missing_market_probability=missing_market_probability,
            missing_opening_odds=missing_opening_odds,
            with_closing=with_closing,
            with_xg=with_xg,
            with_elo=with_elo,
            with_form=with_form,
            training_ready=ready_count,
            total_rows=int(total_rows),
        )


def rebuild_football_dataset_v15(
    settings: Settings,
) -> FootballDatasetV15Summary:
    return FootballDatasetV15(settings).rebuild()


if __name__ == "__main__":
    settings = Settings.from_env()
    summary = rebuild_football_dataset_v15(settings)

    print(
        "Football Dataset v15: "
        f"discovered={summary.discovered}, "
        f"inserted={summary.inserted}, "
        f"updated={summary.updated}, "
        f"missing_model_probability="
        f"{summary.missing_model_probability}, "
        f"missing_market_probability="
        f"{summary.missing_market_probability}, "
        f"missing_opening_odds="
        f"{summary.missing_opening_odds}, "
        f"with_closing={summary.with_closing}, "
        f"with_xg={summary.with_xg}, "
        f"with_elo={summary.with_elo}, "
        f"with_form={summary.with_form}, "
        f"training_ready={summary.training_ready}, "
        f"total={summary.total_rows}"
    )
