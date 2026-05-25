from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from core.config import Settings


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path(settings: Settings) -> Path:
    return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))


def _connect(settings: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(settings))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _norm(name: str) -> str:
    return " ".join(str(name).lower().strip().split())


def _winner_from_scores(event: dict[str, Any]) -> str | None:
    scores = event.get("scores") or []
    if len(scores) < 2:
        return None

    parsed = []
    for item in scores:
        name = str(item.get("name", ""))
        score_raw = item.get("score", None)

        try:
            score = int(score_raw)
        except Exception:
            try:
                score = int(float(score_raw))
            except Exception:
                return None

        parsed.append((name, score))

    if len(parsed) < 2:
        return None

    if parsed[0][1] > parsed[1][1]:
        return parsed[0][0]

    if parsed[1][1] > parsed[0][1]:
        return parsed[1][0]

    return "DRAW"


async def fetch_scores(api_key: str, sport_key: str, days_from: int = 3) -> list[dict[str, Any]]:
    if not api_key:
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/"
    params = {
        "apiKey": api_key,
        "daysFrom": max(1, min(int(days_from), 3)),
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"WARNING | Scores API {resp.status} for {sport_key}: {body[:200]}")
                    return []

                data = await resp.json()
                return data if isinstance(data, list) else []

    except Exception as exc:
        print(f"WARNING | Scores API error for {sport_key}: {exc}")
        return []


async def settle_sport_bets(settings: Settings, sport: str, sport_keys: list[str]) -> int:
    """
    Settlement pre tennis/hockey sport_bets.

    Funguje pre h2h market:
    - výhra selection == víťaz zo scores endpointu
    - pre draw vracia DRAW, ale tenis/hokej h2h väčšinou draw nemajú
    """

    api_key = settings.odds_api_key or os.getenv("ODDS_API_KEY", "")
    if not api_key:
        return 0

    settled = 0

    with _connect(settings) as conn:
        open_rows = conn.execute("""
            SELECT id, league, event, home_team, away_team, selection
            FROM sport_bets
            WHERE sport=?
              AND market='h2h'
              AND (result IS NULL OR result='')
        """, (sport,)).fetchall()

    if not open_rows:
        return 0

    rows_by_league: dict[str, list[tuple[Any, ...]]] = {}
    for row in open_rows:
        rows_by_league.setdefault(str(row[1]), []).append(row)

    for sport_key in sport_keys:
        sport_key = sport_key.strip()
        if not sport_key:
            continue

        scores = await fetch_scores(api_key, sport_key, days_from=3)
        if not scores:
            continue

        score_events = []
        for event in scores:
            if not event.get("completed"):
                continue

            home = str(event.get("home_team", ""))
            away = str(event.get("away_team", ""))
            winner = _winner_from_scores(event)

            if not winner:
                continue

            score_events.append({
                "home": home,
                "away": away,
                "event": f"{home} vs {away}",
                "winner": winner,
            })

        if not score_events:
            continue

        updates: list[tuple[str, str, int]] = []

        for row in rows_by_league.get(sport_key, []):
            bet_id, league, event_name, home_team, away_team, selection = row

            bet_home = _norm(home_team or "")
            bet_away = _norm(away_team or "")
            bet_event = _norm(event_name or "")
            bet_selection = _norm(selection or "")

            matched = None

            for score_event in score_events:
                score_home = _norm(score_event["home"])
                score_away = _norm(score_event["away"])
                score_event_name = _norm(score_event["event"])

                if bet_home and bet_away:
                    if bet_home == score_home and bet_away == score_away:
                        matched = score_event
                        break

                if bet_event and bet_event == score_event_name:
                    matched = score_event
                    break

            if not matched:
                continue

            winner = str(matched["winner"])
            won = _norm(winner) == bet_selection

            updates.append(("V" if won else "P", now_utc(), int(bet_id)))

        if updates:
            with _connect(settings) as conn:
                conn.executemany(
                    "UPDATE sport_bets SET result=?, settled_at=? WHERE id=?",
                    updates,
                )

            settled += len(updates)

    return settled
