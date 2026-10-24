from __future__ import annotations

import csv
import io
import os
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


NFLVERSE_SCHEDULE_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

TEAM_ABBREVIATIONS = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


@dataclass(frozen=True)
class NFLVerseSignal:
    adjustment: float = 0.0
    home_form_margin: float = 0.0
    away_form_margin: float = 0.0
    home_rest: int | None = None
    away_rest: int | None = None
    home_games: int = 0
    away_games: int = 0
    available: bool = False
    reason: str = "nflverse unavailable"


def _parse_number(value: str | None) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_schedule_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def refresh_schedule_cache(
    cache_path: Path,
    *,
    max_age_hours: float = 18.0,
    timeout: int = 20,
) -> tuple[Path | None, str]:
    """Refresh the public NFLverse schedule, keeping failures non-fatal."""
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_hours <= max_age_hours:
            return cache_path, "cached"

    if os.getenv("NFLVERSE_CONTEXT_ENABLED", "1") != "1":
        return (cache_path if cache_path.exists() else None), "disabled"

    try:
        request = urllib.request.Request(
            os.getenv("NFLVERSE_SCHEDULE_URL", NFLVERSE_SCHEDULE_URL),
            headers={"User-Agent": "FutbalDnes/2.0 NFL context"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        if not payload.startswith(b"game_id,season,"):
            raise ValueError("unexpected NFLverse schedule format")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(cache_path)
        return cache_path, "refreshed"
    except Exception as exc:
        if cache_path.exists():
            return cache_path, f"stale cache: {type(exc).__name__}"
        return None, f"download failed: {type(exc).__name__}"


def _team_margins(
    rows: list[dict[str, str]],
    team: str,
    before: datetime,
    limit: int = 8,
) -> list[float]:
    completed: list[tuple[str, float]] = []
    for row in rows:
        gameday = str(row.get("gameday", ""))
        if not gameday:
            continue
        try:
            game_date = datetime.fromisoformat(gameday).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if game_date >= before:
            continue
        home_score = _parse_number(row.get("home_score"))
        away_score = _parse_number(row.get("away_score"))
        if home_score is None or away_score is None:
            continue
        if row.get("home_team") == team:
            margin = home_score - away_score
        elif row.get("away_team") == team:
            margin = away_score - home_score
        else:
            continue
        completed.append((gameday, margin))
    completed.sort(key=lambda item: item[0], reverse=True)
    return [margin for _, margin in completed[:limit]]


def build_signal(
    rows: list[dict[str, str]],
    *,
    home_name: str,
    away_name: str,
    selection: str,
    commence_time: str,
) -> NFLVerseSignal:
    home = TEAM_ABBREVIATIONS.get(home_name, home_name)
    away = TEAM_ABBREVIATIONS.get(away_name, away_name)
    try:
        before = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        if before.tzinfo is None:
            before = before.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        before = datetime.now(timezone.utc)

    home_margins = _team_margins(rows, home, before)
    away_margins = _team_margins(rows, away, before)
    if len(home_margins) < 3 or len(away_margins) < 3:
        return NFLVerseSignal(
            home_games=len(home_margins),
            away_games=len(away_margins),
            reason="insufficient NFLverse history",
        )

    home_form = sum(home_margins) / len(home_margins)
    away_form = sum(away_margins) / len(away_margins)
    home_rest: int | None = None
    away_rest: int | None = None
    event_date = before.date().isoformat()
    for row in rows:
        if (
            row.get("gameday") == event_date
            and row.get("home_team") == home
            and row.get("away_team") == away
        ):
            home_rest_value = _parse_number(row.get("home_rest"))
            away_rest_value = _parse_number(row.get("away_rest"))
            home_rest = int(home_rest_value) if home_rest_value is not None else None
            away_rest = int(away_rest_value) if away_rest_value is not None else None
            break

    form_adjustment = max(-0.035, min(0.035, (home_form - away_form) / 400.0))
    rest_adjustment = 0.0
    if home_rest is not None and away_rest is not None:
        rest_adjustment = max(-0.008, min(0.008, (home_rest - away_rest) * 0.002))
    home_adjustment = max(-0.04, min(0.04, form_adjustment + rest_adjustment))

    if selection == home_name:
        adjustment = home_adjustment
    elif selection == away_name:
        adjustment = -home_adjustment
    else:
        adjustment = 0.0

    return NFLVerseSignal(
        adjustment=adjustment,
        home_form_margin=home_form,
        away_form_margin=away_form,
        home_rest=home_rest,
        away_rest=away_rest,
        home_games=len(home_margins),
        away_games=len(away_margins),
        available=True,
        reason=(
            f"form_margin={home_form:.2f}/{away_form:.2f}; "
            f"rest={home_rest}/{away_rest}; adjustment={adjustment:+.4f}"
        ),
    )

