from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.football_elo import FootballEloDatabase
from core.football_team_form import FootballFormDatabase
from core.football_xg import FootballXGDatabase


SCORE_COLUMN_PAIRS = (
    ("home_goals", "away_goals"),
    ("home_score", "away_score"),
    ("score_home", "score_away"),
    ("goals_home", "goals_away"),
)

XG_COLUMN_PAIRS = (
    ("home_xg", "away_xg"),
    ("xg_home", "xg_away"),
    ("home_expected_goals", "away_expected_goals"),
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_result(value: Any) -> str:
    result = str(value or "").strip().upper()

    if result == "V":
        return "WON"

    if result == "P":
        return "LOST"

    return result


def make_hash(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


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


def parse_score_text(value: Any) -> tuple[int, int] | None:
    text = str(value or "").strip()

    if not text:
        return None

    match = re.search(r"(\d+)\s*[-:]\s*(\d+)", text)

    if match is None:
        return None

    return int(match.group(1)), int(match.group(2))


@dataclass
class SettledFootballMatch:
    league: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int

    home_xg: float | None
    away_xg: float | None

    played_at: str
    source_hash: str
    source: str = "sport_bets"


@dataclass
class FootballResultLearningSummary:
    discovered: int
    processed: int
    skipped_without_score: int
    xg_updates: int
    elo_updates: int
    form_updates: int


class FootballResultLearning:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")

        self.xg = FootballXGDatabase(settings)
        self.elo = FootballEloDatabase(settings)
        self.form = FootballFormDatabase(settings)

        self.xg.init_db()
        self.elo.init_db()
        self.form.init_db()
        self._init_state_table()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_state_table(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_result_learning_state (
                    source_hash TEXT PRIMARY KEY,
                    league TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_goals INTEGER NOT NULL,
                    away_goals INTEGER NOT NULL,
                    processed_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

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

    def _already_processed(self, source_hash: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM football_result_learning_state
                WHERE source_hash=?
                """,
                (source_hash,),
            ).fetchone()

        return row is not None

    def _mark_processed(self, match: SettledFootballMatch) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO football_result_learning_state (
                    source_hash,
                    league,
                    home_team,
                    away_team,
                    home_goals,
                    away_goals,
                    processed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match.source_hash,
                    match.league,
                    match.home_team,
                    match.away_team,
                    match.home_goals,
                    match.away_goals,
                    now_utc(),
                ),
            )
            conn.commit()

    def _extract_score(
        self,
        row: sqlite3.Row,
        columns: set[str],
    ) -> tuple[int, int] | None:
        for home_column, away_column in SCORE_COLUMN_PAIRS:
            if home_column in columns and away_column in columns:
                home = safe_int(row[home_column])
                away = safe_int(row[away_column])

                if home is not None and away is not None:
                    return home, away

        for candidate in (
            "final_score",
            "score",
            "result_score",
            "match_score",
        ):
            if candidate in columns:
                parsed = parse_score_text(row[candidate])

                if parsed is not None:
                    return parsed

        return None

    def _extract_xg(
        self,
        row: sqlite3.Row,
        columns: set[str],
        score: tuple[int, int],
    ) -> tuple[float | None, float | None]:
        for home_column, away_column in XG_COLUMN_PAIRS:
            if home_column in columns and away_column in columns:
                home_xg = safe_float(row[home_column])
                away_xg = safe_float(row[away_column])

                if home_xg is not None and away_xg is not None:
                    return home_xg, away_xg

        allow_goal_proxy = (
            os.getenv("FOOTBALL_ALLOW_GOALS_AS_XG", "0") == "1"
        )

        if allow_goal_proxy:
            return float(score[0]), float(score[1])

        return None, None

    def discover_matches(self) -> tuple[list[SettledFootballMatch], int]:
        with self.connect() as conn:
            if not self._table_exists(conn, "sport_bets"):
                return [], 0

            columns = self._columns(conn, "sport_bets")

            required = {"sport", "league", "event", "result"}

            if not required.issubset(columns):
                return [], 0

            rows = conn.execute(
                """
                SELECT *
                FROM sport_bets
                WHERE sport='football'
                  AND UPPER(TRIM(result))
                      IN ('WON','LOST','VOID','V','P')
                ORDER BY id
                """
            ).fetchall()

        matches: list[SettledFootballMatch] = []
        skipped_without_score = 0

        for row in rows:
            teams = split_event(row["event"])

            if teams is None:
                skipped_without_score += 1
                continue

            score = self._extract_score(row, columns)

            if score is None:
                skipped_without_score += 1
                continue

            home_team, away_team = teams
            home_goals, away_goals = score

            home_xg, away_xg = self._extract_xg(
                row,
                columns,
                score,
            )

            played_at = (
                normalize_text(row["settled_at"])
                if "settled_at" in columns and row["settled_at"]
                else (
                    normalize_text(row["start_time"])
                    if "start_time" in columns
                    else now_utc()
                )
            )

            source_hash = (
                normalize_text(row["source_hash"])
                if "source_hash" in columns and row["source_hash"]
                else make_hash(
                    row["league"],
                    home_team,
                    away_team,
                    played_at,
                    home_goals,
                    away_goals,
                )
            )

            matches.append(
                SettledFootballMatch(
                    league=normalize_text(row["league"]) or "UNKNOWN",
                    home_team=home_team,
                    away_team=away_team,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    home_xg=home_xg,
                    away_xg=away_xg,
                    played_at=played_at,
                    source_hash=source_hash,
                )
            )

        return matches, skipped_without_score

    def run(self) -> FootballResultLearningSummary:
        matches, skipped_without_score = self.discover_matches()

        processed = 0
        xg_updates = 0
        elo_updates = 0
        form_updates = 0

        for match in matches:
            if self._already_processed(match.source_hash):
                continue

            elo_result = self.elo.update_after_match(
                league=match.league,
                home_team=match.home_team,
                away_team=match.away_team,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                played_at=match.played_at,
                source=match.source,
                source_hash=f"{match.source_hash}:elo",
            )

            if elo_result.inserted:
                elo_updates += 1

            form_inserted = self.form.update_after_match(
                league=match.league,
                home_team=match.home_team,
                away_team=match.away_team,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                home_xg=match.home_xg,
                away_xg=match.away_xg,
                played_at=match.played_at,
                source=match.source,
                source_hash=f"{match.source_hash}:form",
            )

            if form_inserted:
                form_updates += 1

            if match.home_xg is not None and match.away_xg is not None:
                xg_inserted = self.xg.update_after_match(
                    league=match.league,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    home_xg=match.home_xg,
                    away_xg=match.away_xg,
                    home_goals=match.home_goals,
                    away_goals=match.away_goals,
                    played_at=match.played_at,
                    source=match.source,
                    source_hash=f"{match.source_hash}:xg",
                )

                if xg_inserted:
                    xg_updates += 1

            self._mark_processed(match)
            processed += 1

        return FootballResultLearningSummary(
            discovered=len(matches),
            processed=processed,
            skipped_without_score=skipped_without_score,
            xg_updates=xg_updates,
            elo_updates=elo_updates,
            form_updates=form_updates,
        )


def run_football_result_learning(
    settings: Settings,
) -> FootballResultLearningSummary:
    return FootballResultLearning(settings).run()


if __name__ == "__main__":
    settings = Settings.from_env()
    summary = run_football_result_learning(settings)

    print(
        "Football result learning: "
        f"discovered={summary.discovered}, "
        f"processed={summary.processed}, "
        f"missing_score={summary.skipped_without_score}, "
        f"xg={summary.xg_updates}, "
        f"elo={summary.elo_updates}, "
        f"form={summary.form_updates}"
    )
