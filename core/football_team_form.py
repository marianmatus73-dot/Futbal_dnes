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


DEFAULT_FORM = 0.50
DEFAULT_GOALS = 1.35


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
class TeamForm:
    team: str
    league: str = "UNKNOWN"

    matches: int = 0

    points_last_5: int = 0
    points_last_10: int = 0

    goals_for_last_5: float = 0.0
    goals_against_last_5: float = 0.0
    goals_for_last_10: float = 0.0
    goals_against_last_10: float = 0.0

    xg_for_last_5: float = DEFAULT_GOALS
    xga_last_5: float = DEFAULT_GOALS
    xg_for_last_10: float = DEFAULT_GOALS
    xga_last_10: float = DEFAULT_GOALS

    home_points_last_5: int = 0
    away_points_last_5: int = 0

    wins_last_5: int = 0
    draws_last_5: int = 0
    losses_last_5: int = 0

    wins_last_10: int = 0
    draws_last_10: int = 0
    losses_last_10: int = 0

    form_score: float = DEFAULT_FORM
    attack_form: float = 1.0
    defense_form: float = 1.0

    last_result: str = ""
    last_updated: str = ""

    def normalized(self) -> "TeamForm":
        self.team = normalize_team_name(self.team)
        self.league = str(self.league or "UNKNOWN").strip() or "UNKNOWN"

        self.matches = max(0, safe_int(self.matches))

        for field_name in (
            "points_last_5",
            "points_last_10",
            "home_points_last_5",
            "away_points_last_5",
            "wins_last_5",
            "draws_last_5",
            "losses_last_5",
            "wins_last_10",
            "draws_last_10",
            "losses_last_10",
        ):
            setattr(
                self,
                field_name,
                max(0, safe_int(getattr(self, field_name))),
            )

        for field_name in (
            "goals_for_last_5",
            "goals_against_last_5",
            "goals_for_last_10",
            "goals_against_last_10",
            "xg_for_last_5",
            "xga_last_5",
            "xg_for_last_10",
            "xga_last_10",
        ):
            setattr(
                self,
                field_name,
                clamp(
                    safe_float(getattr(self, field_name), DEFAULT_GOALS),
                    0.0,
                    6.0,
                ),
            )

        self.form_score = clamp(
            safe_float(self.form_score, DEFAULT_FORM),
            0.0,
            1.0,
        )
        self.attack_form = clamp(
            safe_float(self.attack_form, 1.0),
            0.40,
            2.50,
        )
        self.defense_form = clamp(
            safe_float(self.defense_form, 1.0),
            0.40,
            2.50,
        )

        if not self.last_updated:
            self.last_updated = now_utc()

        return self

    @property
    def reliability(self) -> float:
        return clamp(self.matches / 10.0, 0.0, 1.0)


@dataclass
class FormPrediction:
    home_team: str
    away_team: str
    league: str

    home_form_probability: float
    draw_form_probability: float
    away_form_probability: float

    home_form_score: float
    away_form_score: float

    home_attack_multiplier: float
    home_defense_multiplier: float
    away_attack_multiplier: float
    away_defense_multiplier: float

    home_reliability: float
    away_reliability: float

    reason: str


class FootballFormDatabase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_team_form (
                    team TEXT NOT NULL,
                    league TEXT NOT NULL,

                    matches INTEGER NOT NULL DEFAULT 0,

                    points_last_5 INTEGER NOT NULL DEFAULT 0,
                    points_last_10 INTEGER NOT NULL DEFAULT 0,

                    goals_for_last_5 REAL NOT NULL DEFAULT 0.0,
                    goals_against_last_5 REAL NOT NULL DEFAULT 0.0,
                    goals_for_last_10 REAL NOT NULL DEFAULT 0.0,
                    goals_against_last_10 REAL NOT NULL DEFAULT 0.0,

                    xg_for_last_5 REAL NOT NULL DEFAULT 1.35,
                    xga_last_5 REAL NOT NULL DEFAULT 1.35,
                    xg_for_last_10 REAL NOT NULL DEFAULT 1.35,
                    xga_last_10 REAL NOT NULL DEFAULT 1.35,

                    home_points_last_5 INTEGER NOT NULL DEFAULT 0,
                    away_points_last_5 INTEGER NOT NULL DEFAULT 0,

                    wins_last_5 INTEGER NOT NULL DEFAULT 0,
                    draws_last_5 INTEGER NOT NULL DEFAULT 0,
                    losses_last_5 INTEGER NOT NULL DEFAULT 0,

                    wins_last_10 INTEGER NOT NULL DEFAULT 0,
                    draws_last_10 INTEGER NOT NULL DEFAULT 0,
                    losses_last_10 INTEGER NOT NULL DEFAULT 0,

                    form_score REAL NOT NULL DEFAULT 0.5,
                    attack_form REAL NOT NULL DEFAULT 1.0,
                    defense_form REAL NOT NULL DEFAULT 1.0,

                    last_result TEXT NOT NULL DEFAULT '',
                    last_updated TEXT NOT NULL,

                    PRIMARY KEY (team, league)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_form_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    played_at TEXT NOT NULL,
                    league TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_goals INTEGER NOT NULL,
                    away_goals INTEGER NOT NULL,
                    home_xg REAL,
                    away_xg REAL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    source_hash TEXT UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.commit()

    def load_team(self, team: str, league: str = "UNKNOWN") -> TeamForm:
        self.init_db()

        team = normalize_team_name(team)
        league = str(league or "UNKNOWN").strip() or "UNKNOWN"

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM football_team_form
                WHERE team=? AND league=?
                """,
                (team, league),
            ).fetchone()

        if row is None:
            return TeamForm(
                team=team,
                league=league,
                last_updated=now_utc(),
            ).normalized()

        return TeamForm(**dict(row)).normalized()

    def save_team(self, form: TeamForm) -> None:
        self.init_db()
        form = form.normalized()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO football_team_form (
                    team,
                    league,
                    matches,
                    points_last_5,
                    points_last_10,
                    goals_for_last_5,
                    goals_against_last_5,
                    goals_for_last_10,
                    goals_against_last_10,
                    xg_for_last_5,
                    xga_last_5,
                    xg_for_last_10,
                    xga_last_10,
                    home_points_last_5,
                    away_points_last_5,
                    wins_last_5,
                    draws_last_5,
                    losses_last_5,
                    wins_last_10,
                    draws_last_10,
                    losses_last_10,
                    form_score,
                    attack_form,
                    defense_form,
                    last_result,
                    last_updated
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(team, league) DO UPDATE SET
                    matches=excluded.matches,
                    points_last_5=excluded.points_last_5,
                    points_last_10=excluded.points_last_10,
                    goals_for_last_5=excluded.goals_for_last_5,
                    goals_against_last_5=excluded.goals_against_last_5,
                    goals_for_last_10=excluded.goals_for_last_10,
                    goals_against_last_10=excluded.goals_against_last_10,
                    xg_for_last_5=excluded.xg_for_last_5,
                    xga_last_5=excluded.xga_last_5,
                    xg_for_last_10=excluded.xg_for_last_10,
                    xga_last_10=excluded.xga_last_10,
                    home_points_last_5=excluded.home_points_last_5,
                    away_points_last_5=excluded.away_points_last_5,
                    wins_last_5=excluded.wins_last_5,
                    draws_last_5=excluded.draws_last_5,
                    losses_last_5=excluded.losses_last_5,
                    wins_last_10=excluded.wins_last_10,
                    draws_last_10=excluded.draws_last_10,
                    losses_last_10=excluded.losses_last_10,
                    form_score=excluded.form_score,
                    attack_form=excluded.attack_form,
                    defense_form=excluded.defense_form,
                    last_result=excluded.last_result,
                    last_updated=excluded.last_updated
                """,
                (
                    form.team,
                    form.league,
                    form.matches,
                    form.points_last_5,
                    form.points_last_10,
                    form.goals_for_last_5,
                    form.goals_against_last_5,
                    form.goals_for_last_10,
                    form.goals_against_last_10,
                    form.xg_for_last_5,
                    form.xga_last_5,
                    form.xg_for_last_10,
                    form.xga_last_10,
                    form.home_points_last_5,
                    form.away_points_last_5,
                    form.wins_last_5,
                    form.draws_last_5,
                    form.losses_last_5,
                    form.wins_last_10,
                    form.draws_last_10,
                    form.losses_last_10,
                    form.form_score,
                    form.attack_form,
                    form.defense_form,
                    form.last_result,
                    form.last_updated,
                ),
            )
            conn.commit()

    def _recent_matches(
        self,
        team: str,
        league: str,
        limit: int = 10,
    ) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT *
                FROM football_form_history
                WHERE league=?
                  AND (home_team=? OR away_team=?)
                ORDER BY played_at DESC, id DESC
                LIMIT ?
                """,
                (league, team, team, max(1, limit)),
            ).fetchall()

    def rebuild_team(self, team: str, league: str) -> TeamForm:
        self.init_db()

        team = normalize_team_name(team)
        league = str(league or "UNKNOWN").strip() or "UNKNOWN"
        rows = self._recent_matches(team, league, limit=10)

        form = TeamForm(team=team, league=league)
        form.matches = len(rows)

        points_5 = 0
        points_10 = 0

        wins_5 = draws_5 = losses_5 = 0
        wins_10 = draws_10 = losses_10 = 0

        gf_5 = ga_5 = gf_10 = ga_10 = 0.0
        xgf_5 = xga_5 = xgf_10 = xga_10 = 0.0

        home_points_5 = 0
        away_points_5 = 0

        for index, row in enumerate(rows):
            is_home = row["home_team"] == team

            goals_for = (
                safe_int(row["home_goals"])
                if is_home
                else safe_int(row["away_goals"])
            )
            goals_against = (
                safe_int(row["away_goals"])
                if is_home
                else safe_int(row["home_goals"])
            )

            xg_for = (
                safe_float(row["home_xg"], goals_for)
                if is_home
                else safe_float(row["away_xg"], goals_for)
            )
            xg_against = (
                safe_float(row["away_xg"], goals_against)
                if is_home
                else safe_float(row["home_xg"], goals_against)
            )

            if goals_for > goals_against:
                points = 3
                result = "W"
            elif goals_for == goals_against:
                points = 1
                result = "D"
            else:
                points = 0
                result = "L"

            points_10 += points
            gf_10 += goals_for
            ga_10 += goals_against
            xgf_10 += xg_for
            xga_10 += xg_against

            if result == "W":
                wins_10 += 1
            elif result == "D":
                draws_10 += 1
            else:
                losses_10 += 1

            if index < 5:
                points_5 += points
                gf_5 += goals_for
                ga_5 += goals_against
                xgf_5 += xg_for
                xga_5 += xg_against

                if result == "W":
                    wins_5 += 1
                elif result == "D":
                    draws_5 += 1
                else:
                    losses_5 += 1

                if is_home:
                    home_points_5 += points
                else:
                    away_points_5 += points

            if index == 0:
                form.last_result = result

        count_5 = min(5, len(rows))
        count_10 = max(1, len(rows))

        form.points_last_5 = points_5
        form.points_last_10 = points_10

        form.goals_for_last_5 = gf_5 / max(1, count_5)
        form.goals_against_last_5 = ga_5 / max(1, count_5)
        form.goals_for_last_10 = gf_10 / count_10
        form.goals_against_last_10 = ga_10 / count_10

        form.xg_for_last_5 = xgf_5 / max(1, count_5)
        form.xga_last_5 = xga_5 / max(1, count_5)
        form.xg_for_last_10 = xgf_10 / count_10
        form.xga_last_10 = xga_10 / count_10

        form.home_points_last_5 = home_points_5
        form.away_points_last_5 = away_points_5

        form.wins_last_5 = wins_5
        form.draws_last_5 = draws_5
        form.losses_last_5 = losses_5

        form.wins_last_10 = wins_10
        form.draws_last_10 = draws_10
        form.losses_last_10 = losses_10

        points_component = (
            (points_5 / max(1, count_5 * 3)) * 0.60
            + (points_10 / max(1, count_10 * 3)) * 0.40
        )

        attack_component = (
            (form.xg_for_last_5 / DEFAULT_GOALS) * 0.55
            + (form.xg_for_last_10 / DEFAULT_GOALS) * 0.45
        )

        defense_component = (
            (form.xga_last_5 / DEFAULT_GOALS) * 0.55
            + (form.xga_last_10 / DEFAULT_GOALS) * 0.45
        )

        form.form_score = clamp(points_component, 0.0, 1.0)
        form.attack_form = clamp(attack_component, 0.40, 2.50)
        form.defense_form = clamp(defense_component, 0.40, 2.50)
        form.last_updated = now_utc()

        self.save_team(form)
        return form

    def update_after_match(
        self,
        *,
        league: str,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        home_xg: float | None = None,
        away_xg: float | None = None,
        played_at: str | None = None,
        source: str = "manual",
        source_hash: str | None = None,
    ) -> bool:
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
            before = conn.total_changes

            conn.execute(
                """
                INSERT OR IGNORE INTO football_form_history (
                    played_at,
                    league,
                    home_team,
                    away_team,
                    home_goals,
                    away_goals,
                    home_xg,
                    away_xg,
                    source,
                    source_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    played_at,
                    league,
                    home_team,
                    away_team,
                    home_goals,
                    away_goals,
                    home_xg,
                    away_xg,
                    source,
                    source_hash,
                    now_utc(),
                ),
            )

            inserted = conn.total_changes > before
            conn.commit()

        if not inserted:
            return False

        self.rebuild_team(home_team, league)
        self.rebuild_team(away_team, league)
        return True

    def predict_match(
        self,
        *,
        home_team: str,
        away_team: str,
        league: str,
    ) -> FormPrediction:
        home = self.load_team(home_team, league)
        away = self.load_team(away_team, league)

        home_strength = (
            home.form_score * 0.55
            + clamp(home.attack_form / 2.0, 0.0, 1.0) * 0.25
            + clamp(1.0 / home.defense_form, 0.0, 2.0) / 2.0 * 0.20
        )

        away_strength = (
            away.form_score * 0.55
            + clamp(away.attack_form / 2.0, 0.0, 1.0) * 0.25
            + clamp(1.0 / away.defense_form, 0.0, 2.0) / 2.0 * 0.20
        )

        difference = home_strength - away_strength
        home_probability = 0.37 + difference * 0.35
        away_probability = 0.37 - difference * 0.35

        parity = 1.0 - clamp(abs(difference), 0.0, 1.0)
        draw_probability = 0.22 + parity * 0.08

        reliability = (home.reliability + away.reliability) / 2.0

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
            home_probability, draw_probability, away_probability = (
                0.37,
                0.26,
                0.37,
            )
        else:
            home_probability /= total
            draw_probability /= total
            away_probability /= total

        reason = (
            f"Form prediction: home={home_probability:.3f}, "
            f"draw={draw_probability:.3f}, away={away_probability:.3f}; "
            f"home_form={home.form_score:.3f}; "
            f"away_form={away.form_score:.3f}; "
            f"reliability={reliability:.2f}"
        )

        return FormPrediction(
            home_team=home.team,
            away_team=away.team,
            league=league,
            home_form_probability=clamp(home_probability, 0.01, 0.98),
            draw_form_probability=clamp(draw_probability, 0.01, 0.60),
            away_form_probability=clamp(away_probability, 0.01, 0.98),
            home_form_score=home.form_score,
            away_form_score=away.form_score,
            home_attack_multiplier=home.attack_form,
            home_defense_multiplier=home.defense_form,
            away_attack_multiplier=away.attack_form,
            away_defense_multiplier=away.defense_form,
            home_reliability=home.reliability,
            away_reliability=away.reliability,
            reason=reason,
        )

    def export_json(
        self,
        path: str = "exports/football_team_form.json",
    ) -> int:
        self.init_db()

        with self.connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM football_team_form
                    ORDER BY league, form_score DESC
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


def predict_football_form(
    settings: Settings,
    *,
    home_team: str,
    away_team: str,
    league: str,
) -> FormPrediction:
    database = FootballFormDatabase(settings)

    return database.predict_match(
        home_team=home_team,
        away_team=away_team,
        league=league,
    )


def update_football_form(
    settings: Settings,
    **kwargs: Any,
) -> bool:
    database = FootballFormDatabase(settings)
    return database.update_after_match(**kwargs)
