from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import aiohttp

from core.config import Settings
from core.sport_quant import (
    connect,
    norm,
    update_closing_lines,
    refresh_bookmaker_stats,
    update_elo_after_result,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _winner_from_scores(event: dict[str, Any]) -> str | None:
    scores = event.get("scores") or []
    if len(scores) < 2:
        return None

    parsed = []
    for item in scores:
        name = str(item.get("name", ""))
        score_raw = item.get("score")

        try:
            score = int(score_raw)
        except Exception:
            try:
                score = int(float(score_raw))
            except Exception:
                return None

        parsed.append((name, score))

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
    api_key = settings.odds_api_key or os.getenv("ODDS_API_KEY", "")
    if not api_key:
        return 0

    update_closing_lines(settings, sport)

    with connect(settings) as conn:
        open_rows = conn.execute("""
            SELECT id, league, event, home_team, away_team, selection
            FROM sport_bets
            WHERE sport=?
              AND market='h2h'
              AND (result IS NULL OR result='')
        """, (sport,)).fetchall()

    if not open_rows:
        refresh_bookmaker_stats(settings, sport)
        return 0

    rows_by_league: dict[str, list[tuple[Any, ...]]] = {}
    for row in open_rows:
        rows_by_league.setdefault(str(row[1]), []).append(row)

    settled = 0

    for sport_key in [s.strip() for s in sport_keys if s.strip()]:
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
        elo_updates: list[tuple[str, str, str]] = []

        for row in rows_by_league.get(sport_key, []):
            bet_id, league, event_name, home_team, away_team, selection = row

            bet_home = norm(home_team or "")
            bet_away = norm(away_team or "")
            bet_event = norm(event_name or "")
            bet_selection = norm(selection or "")

            matched = None

            for score_event in score_events:
                score_home = norm(score_event["home"])
                score_away = norm(score_event["away"])
                score_event_name = norm(score_event["event"])

                if bet_home and bet_away and bet_home == score_home and bet_away == score_away:
                    matched = score_event
                    break

                if bet_event and bet_event == score_event_name:
                    matched = score_event
                    break

            if not matched:
                continue

            winner = str(matched["winner"])
            won = norm(winner) == bet_selection
            updates.append(("V" if won else "P", now_utc(), int(bet_id)))
            elo_updates.append((str(matched["home"]), str(matched["away"]), winner))

        if updates:
            with connect(settings) as conn:
                conn.executemany(
                    "UPDATE sport_bets SET result=?, settled_at=? WHERE id=?",
                    updates,
                )

            settled += len(updates)

            for home, away, winner in elo_updates:
                k = 20.0 if sport == "tennis" else 24.0
                home_adv = 0.0 if sport == "tennis" else float(os.getenv("HOCKEY_HOME_ELO_ADV", "35"))
                update_elo_after_result(settings, sport, home, away, winner, k=k, home_adv=home_adv)

    update_closing_lines(settings, sport)
    refresh_bookmaker_stats(settings, sport)

    return settled
