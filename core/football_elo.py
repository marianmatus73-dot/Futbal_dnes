from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


DEFAULT_ELO = 1500.0
DEFAULT_HOME_ADVANTAGE_ELO = 65.0
DEFAULT_K_FACTOR = 24.0
MIN_ELO = 800.0
MAX_ELO = 2400.0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_team_name(team: str) -> str:
    return " ".join(str(team or "").strip().split())


def make_source_hash(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class TeamElo:
    team: str
    league: str = "UNKNOWN"

    overall_elo: float = DEFAULT_ELO
    home_elo: float = DEFAULT_ELO
    away_elo: float = DEFAULT_ELO
    form_elo: float = DEFAULT_ELO

    matches: int = 0
    home_matches: int = 0
    away_matches: int = 0

    wins: int = 0
    draws: int = 0
    losses: int = 0

    goals_for: int = 0
    goals_against: int = 0

    last_result: str = ""
    last_updated: str = ""

    def normalized(self) -> "TeamElo":
        self.team = normalize_team_name(self.team)
        self.league = str(self.league or "UNKNOWN").strip() or "UNKNOWN"

        self.overall_elo = clamp(
            safe_float(self.overall_elo, DEFAULT_ELO),
            MIN_ELO,
            MAX_ELO,
        )
        self.home_elo = clamp(
            safe_float(self.home_elo, DEFAULT_ELO),
            MIN_ELO,
            MAX_ELO,
        )
        self.away_elo = clamp(
            safe_float(self.away_elo, DEFAULT_ELO),
            MIN_ELO,
            MAX_ELO,
        )
        self.form_elo = clamp(
            safe_float(self.form_elo, DEFAULT_ELO),
            MIN_ELO,
            MAX_ELO,
        )

        self.matches = max(0, safe_int(self.matches))
        self.home_matches = max(0, safe_int(self.home_matches))
        self.away_matches = max(0, safe_int(self.away_matches))

        self.wins = max(0, safe_int(self.wins))
        self.draws = max(0, safe_int(self.draws))
        self.losses = max(0, safe_int(self.losses))

        self.goals_for = max(0, safe_int(self.goals_for))
        self.goals_against = max(0, safe_int(self.goals_against))

        if not self.last_updated:
            self.last_updated = now_utc()

        return self

    @property
    def sample_reliability(self) -> float:
        return clamp(self.matches / 30.0, 0.0, 1.0)

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def points_per_match(self) -> float:
        if self.matches <= 0:
            return 0.0

        return (self.wins * 3 + self.draws) / self.matches


@dataclass
class EloPrediction:
    home_team: str
    away_team: str
    league: str

    home_probability: float
    draw_probability: float
    away_probability: float

    home_effective_elo: float
    away_effective_elo: float
    elo_difference: float

    home_reliability: float
    away_reliability: float

    reason: str


@dataclass
class EloUpdateResult:
    home_team: str
    away_team: str
    league: str

    home_before: float
    away_before: float
    home_after: float
    away_after: float

    home_delta: float
    away_delta: float

    expected_home_score: float
    actual_home_score: float
    k_factor: float
    margin_multiplier: float

    inserted: bool


class FootballEloDatabase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_elo_ratings (
                    team TEXT NOT NULL,
                    league TEXT NOT NULL,

                    overall_elo REAL NOT NULL DEFAULT 1500.0,
                    home_elo REAL NOT NULL DEFAULT 1500.0,
                    away_elo REAL NOT NULL DEFAULT 1500.0,
                    form_elo REAL NOT NULL DEFAULT 1500.0,

                    matches INTEGER NOT NULL DEFAULT 0,
                    home_matches INTEGER NOT NULL DEFAULT 0,
                    away_matches INTEGER NOT NULL DEFAULT 0,

                    wins INTEGER NOT NULL DEFAULT 0,
                    draws INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,

                    goals_for INTEGER NOT NULL DEFAULT 0,
                    goals_against INTEGER NOT NULL DEFAULT 0,

                    last_result TEXT NOT NULL DEFAULT '',
                    last_updated TEXT NOT NULL,

                    PRIMARY KEY (team, league)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_elo_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    played_at TEXT NOT NULL,
                    league TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,

                    home_goals INTEGER NOT NULL,
                    away_goals INTEGER NOT NULL,

                    home_elo_before REAL NOT NULL,
                    away_elo_before REAL NOT NULL,
                    home_elo_after REAL NOT NULL,
                    away_elo_after REAL NOT NULL,

                    home_delta REAL NOT NULL,
                    away_delta REAL NOT NULL,

                    expected_home_score REAL NOT NULL,
                    actual_home_score REAL NOT NULL,

                    k_factor REAL NOT NULL,
                    margin_multiplier REAL NOT NULL,
                    importance REAL NOT NULL DEFAULT 1.0,

                    source TEXT NOT NULL DEFAULT 'manual',
                    source_hash TEXT UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_elo_history_match
                ON football_elo_history (
                    league,
                    home_team,
                    away_team,
                    played_at
                )
                """
            )

            conn.commit()

    def load_team(self, team: str, league: str = "UNKNOWN") -> TeamElo:
        self.init_db()

        team = normalize_team_name(team)
        league = str(league or "UNKNOWN").strip() or "UNKNOWN"

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM football_elo_ratings
                WHERE team=? AND league=?
                """,
                (team, league),
            ).fetchone()

        if row is None:
            return TeamElo(
                team=team,
                league=league,
                last_updated=now_utc(),
            ).normalized()

        return TeamElo(**dict(row)).normalized()

    def save_team(self, rating: TeamElo) -> None:
        self.init_db()
        rating = rating.normalized()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO football_elo_ratings (
                    team,
                    league,
                    overall_elo,
                    home_elo,
                    away_elo,
                    form_elo,
                    matches,
                    home_matches,
                    away_matches,
                    wins,
                    draws,
                    losses,
                    goals_for,
                    goals_against,
                    last_result,
                    last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team, league) DO UPDATE SET
                    overall_elo=excluded.overall_elo,
                    home_elo=excluded.home_elo,
                    away_elo=excluded.away_elo,
                    form_elo=excluded.form_elo,
                    matches=excluded.matches,
                    home_matches=excluded.home_matches,
                    away_matches=excluded.away_matches,
                    wins=excluded.wins,
                    draws=excluded.draws,
                    losses=excluded.losses,
                    goals_for=excluded.goals_for,
                    goals_against=excluded.goals_against,
                    last_result=excluded.last_result,
                    last_updated=excluded.last_updated
                """,
                (
                    rating.team,
                    rating.league,
                    rating.overall_elo,
                    rating.home_elo,
                    rating.away_elo,
                    rating.form_elo,
                    rating.matches,
                    rating.home_matches,
                    rating.away_matches,
                    rating.wins,
                    rating.draws,
                    rating.losses,
                    rating.goals_for,
                    rating.goals_against,
                    rating.last_result,
                    rating.last_updated,
                ),
            )
            conn.commit()

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        rating_a = safe_float(rating_a, DEFAULT_ELO)
        rating_b = safe_float(rating_b, DEFAULT_ELO)

        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def margin_multiplier(
        home_goals: int,
        away_goals: int,
        elo_difference: float = 0.0,
    ) -> float:
        margin = abs(int(home_goals) - int(away_goals))

        if margin <= 1:
            base = 1.0
        elif margin == 2:
            base = 1.35
        elif margin == 3:
            base = 1.60
        else:
            base = 1.60 + math.log1p(margin - 3) * 0.35

        correction = 2.2 / (
            2.2 + max(0.0, abs(safe_float(elo_difference, 0.0))) * 0.001
        )

        return clamp(base * correction, 0.80, 2.50)

    @staticmethod
    def dynamic_k_factor(
        home: TeamElo,
        away: TeamElo,
        importance: float = 1.0,
        base_k: float = DEFAULT_K_FACTOR,
    ) -> float:
        importance = clamp(safe_float(importance, 1.0), 0.50, 2.50)
        base_k = clamp(safe_float(base_k, DEFAULT_K_FACTOR), 8.0, 60.0)

        combined_matches = home.matches + away.matches

        if combined_matches < 10:
            sample_multiplier = 1.35
        elif combined_matches < 30:
            sample_multiplier = 1.15
        elif combined_matches > 120:
            sample_multiplier = 0.85
        else:
            sample_multiplier = 1.0

        return clamp(
            base_k * importance * sample_multiplier,
            8.0,
            72.0,
        )

    def predict_match(
        self,
        *,
        home_team: str,
        away_team: str,
        league: str,
        home_advantage_elo: float = DEFAULT_HOME_ADVANTAGE_ELO,
        draw_base: float = 0.26,
    ) -> EloPrediction:
        home = self.load_team(home_team, league)
        away = self.load_team(away_team, league)

        home_advantage_elo = clamp(
            safe_float(home_advantage_elo, DEFAULT_HOME_ADVANTAGE_ELO),
            0.0,
            150.0,
        )
        draw_base = clamp(safe_float(draw_base, 0.26), 0.12, 0.38)

        home_reliability = home.sample_reliability
        away_reliability = away.sample_reliability

        home_effective_elo = (
            home.overall_elo * 0.55
            + home.home_elo * 0.25
            + home.form_elo * 0.20
            + home_advantage_elo
        )

        away_effective_elo = (
            away.overall_elo * 0.55
            + away.away_elo * 0.25
            + away.form_elo * 0.20
        )

        elo_difference = home_effective_elo - away_effective_elo

        two_way_home = self.expected_score(
            home_effective_elo,
            away_effective_elo,
        )

        parity = math.exp(-abs(elo_difference) / 260.0)
        draw_probability = draw_base * (
            0.70 + 0.45 * parity
        )
        draw_probability = clamp(draw_probability, 0.12, 0.38)

        non_draw_probability = 1.0 - draw_probability
        home_probability = two_way_home * non_draw_probability
        away_probability = (1.0 - two_way_home) * non_draw_probability

        reliability = (home_reliability + away_reliability) / 2.0

        home_probability = (
            home_probability * (0.55 + 0.40 * reliability)
            + 0.37 * (0.45 - 0.40 * reliability)
        )
        draw_probability = (
            draw_probability * (0.55 + 0.40 * reliability)
            + 0.26 * (0.45 - 0.40 * reliability)
        )
        away_probability = (
            away_probability * (0.55 + 0.40 * reliability)
            + 0.37 * (0.45 - 0.40 * reliability)
        )

        total = home_probability + draw_probability + away_probability

        if total <= 0:
            home_probability = 0.37
            draw_probability = 0.26
            away_probability = 0.37
        else:
            home_probability /= total
            draw_probability /= total
            away_probability /= total

        reason = (
            f"ELO prediction: home={home_probability:.3f}, "
            f"draw={draw_probability:.3f}, away={away_probability:.3f}; "
            f"home_elo={home_effective_elo:.1f}; "
            f"away_elo={away_effective_elo:.1f}; "
            f"difference={elo_difference:.1f}; "
            f"reliability={reliability:.2f}"
        )

        return EloPrediction(
            home_team=home.team,
            away_team=away.team,
            league=league,
            home_probability=clamp(home_probability, 0.01, 0.98),
            draw_probability=clamp(draw_probability, 0.01, 0.60),
            away_probability=clamp(away_probability, 0.01, 0.98),
            home_effective_elo=home_effective_elo,
            away_effective_elo=away_effective_elo,
            elo_difference=elo_difference,
            home_reliability=home_reliability,
            away_reliability=away_reliability,
            reason=reason,
        )

    def update_after_match(
        self,
        *,
        league: str,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        played_at: str | None = None,
        importance: float = 1.0,
        home_advantage_elo: float = DEFAULT_HOME_ADVANTAGE_ELO,
        source: str = "manual",
        source_hash: str | None = None,
        base_k: float = DEFAULT_K_FACTOR,
    ) -> EloUpdateResult:
        self.init_db()

        league = str(league or "UNKNOWN").strip() or "UNKNOWN"
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)
        home_goals = max(0, safe_int(home_goals))
        away_goals = max(0, safe_int(away_goals))
        played_at = played_at or now_utc()

        source_hash = source_hash or make_source_hash(
            league,
            home_team,
            away_team,
            played_at,
            home_goals,
            away_goals,
        )

        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT 1
                FROM football_elo_history
                WHERE source_hash=?
                """,
                (source_hash,),
            ).fetchone()

        home = self.load_team(home_team, league)
        away = self.load_team(away_team, league)

        if existing:
            return EloUpdateResult(
                home_team=home_team,
                away_team=away_team,
                league=league,
                home_before=home.overall_elo,
                away_before=away.overall_elo,
                home_after=home.overall_elo,
                away_after=away.overall_elo,
                home_delta=0.0,
                away_delta=0.0,
                expected_home_score=0.5,
                actual_home_score=0.5,
                k_factor=0.0,
                margin_multiplier=1.0,
                inserted=False,
            )

        home_before = home.overall_elo
        away_before = away.overall_elo

        home_effective = home.overall_elo + clamp(
            safe_float(home_advantage_elo, DEFAULT_HOME_ADVANTAGE_ELO),
            0.0,
            150.0,
        )

        expected_home = self.expected_score(
            home_effective,
            away.overall_elo,
        )

        if home_goals > away_goals:
            actual_home = 1.0
            home_result = "W"
            away_result = "L"
        elif home_goals < away_goals:
            actual_home = 0.0
            home_result = "L"
            away_result = "W"
        else:
            actual_home = 0.5
            home_result = "D"
            away_result = "D"

        k_factor = self.dynamic_k_factor(
            home,
            away,
            importance=importance,
            base_k=base_k,
        )

        margin = self.margin_multiplier(
            home_goals,
            away_goals,
            elo_difference=home_effective - away.overall_elo,
        )

        home_delta = k_factor * margin * (actual_home - expected_home)
        away_delta = -home_delta

        home.overall_elo = clamp(
            home.overall_elo + home_delta,
            MIN_ELO,
            MAX_ELO,
        )
        away.overall_elo = clamp(
            away.overall_elo + away_delta,
            MIN_ELO,
            MAX_ELO,
        )

        venue_k = k_factor * 0.70
        form_k = k_factor * 0.45

        home.home_elo = clamp(
            home.home_elo
            + venue_k * margin * (actual_home - expected_home),
            MIN_ELO,
            MAX_ELO,
        )
        away.away_elo = clamp(
            away.away_elo
            - venue_k * margin * (actual_home - expected_home),
            MIN_ELO,
            MAX_ELO,
        )

        home.form_elo = clamp(
            home.form_elo
            + form_k * margin * (actual_home - expected_home),
            MIN_ELO,
            MAX_ELO,
        )
        away.form_elo = clamp(
            away.form_elo
            - form_k * margin * (actual_home - expected_home),
            MIN_ELO,
            MAX_ELO,
        )

        home.matches += 1
        home.home_matches += 1
        away.matches += 1
        away.away_matches += 1

        home.goals_for += home_goals
        home.goals_against += away_goals
        away.goals_for += away_goals
        away.goals_against += home_goals

        if home_result == "W":
            home.wins += 1
            away.losses += 1
        elif home_result == "L":
            home.losses += 1
            away.wins += 1
        else:
            home.draws += 1
            away.draws += 1

        home.last_result = home_result
        away.last_result = away_result
        home.last_updated = now_utc()
        away.last_updated = now_utc()

        self.save_team(home)
        self.save_team(away)

        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO football_elo_history (
                    played_at,
                    league,
                    home_team,
                    away_team,
                    home_goals,
                    away_goals,
                    home_elo_before,
                    away_elo_before,
                    home_elo_after,
                    away_elo_after,
                    home_delta,
                    away_delta,
                    expected_home_score,
                    actual_home_score,
                    k_factor,
                    margin_multiplier,
                    importance,
                    source,
                    source_hash,
                    created_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    played_at,
                    league,
                    home_team,
                    away_team,
                    home_goals,
                    away_goals,
                    home_before,
                    away_before,
                    home.overall_elo,
                    away.overall_elo,
                    home_delta,
                    away_delta,
                    expected_home,
                    actual_home,
                    k_factor,
                    margin,
                    importance,
                    source,
                    source_hash,
                    now_utc(),
                ),
            )
            conn.commit()

        return EloUpdateResult(
            home_team=home_team,
            away_team=away_team,
            league=league,
            home_before=home_before,
            away_before=away_before,
            home_after=home.overall_elo,
            away_after=away.overall_elo,
            home_delta=home_delta,
            away_delta=away_delta,
            expected_home_score=expected_home,
            actual_home_score=actual_home,
            k_factor=k_factor,
            margin_multiplier=margin,
            inserted=True,
        )

    def league_table(self, league: str) -> list[TeamElo]:
        self.init_db()

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM football_elo_ratings
                WHERE league=?
                ORDER BY overall_elo DESC
                """,
                (league,),
            ).fetchall()

        return [
            TeamElo(**dict(row)).normalized()
            for row in rows
        ]

    def export_json(
        self,
        path: str = "exports/football_elo_ratings.json",
    ) -> int:
        self.init_db()

        with self.connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM football_elo_ratings
                    ORDER BY league, overall_elo DESC
                    """
                ).fetchall()
            ]

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return len(rows)

    def import_json(
        self,
        path: str = "exports/football_elo_ratings.json",
    ) -> int:
        file_path = Path(path)

        if not file_path.exists():
            return 0

        data = json.loads(file_path.read_text(encoding="utf-8"))

        if not isinstance(data, list):
            return 0

        imported = 0

        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                self.save_team(TeamElo(**item).normalized())
                imported += 1
            except Exception:
                continue

        return imported


def predict_football_elo(
    settings: Settings,
    *,
    home_team: str,
    away_team: str,
    league: str,
    home_advantage_elo: float = DEFAULT_HOME_ADVANTAGE_ELO,
    draw_base: float = 0.26,
) -> EloPrediction:
    database = FootballEloDatabase(settings)

    return database.predict_match(
        home_team=home_team,
        away_team=away_team,
        league=league,
        home_advantage_elo=home_advantage_elo,
        draw_base=draw_base,
    )


def update_football_elo(
    settings: Settings,
    **kwargs: Any,
) -> EloUpdateResult:
    database = FootballEloDatabase(settings)
    return database.update_after_match(**kwargs)
