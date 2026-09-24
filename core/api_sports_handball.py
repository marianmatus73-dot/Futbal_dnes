from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any


BASE_URL = "https://v1.handball.api-sports.io"


class ApiSportsHandballError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApiSportsHandballClient:
    api_key: str
    timeout: int = 30

    def _get_sync(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.api_key:
            raise ApiSportsHandballError("API_SPORTS_KEY is missing")
        query = urllib.parse.urlencode(
            {key: value for key, value in params.items() if value not in (None, "")}
        )
        request = urllib.request.Request(
            f"{BASE_URL}/{endpoint}?{query}",
            headers={"x-apisports-key": self.api_key, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ApiSportsHandballError(
                f"API-Sports Handball HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ApiSportsHandballError(f"API-Sports Handball request failed: {exc}") from exc
        errors = payload.get("errors") or {}
        if errors:
            raise ApiSportsHandballError(f"API-Sports Handball error: {errors}")
        response = payload.get("response") or []
        return response if isinstance(response, list) else []

    async def get(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_sync, endpoint, params)

    async def games_for_date(self, day: date) -> list[dict[str, Any]]:
        return await self.get("games", date=day.isoformat(), timezone="Europe/Bratislava")

    async def odds_for_game(self, game_id: int | str) -> list[dict[str, Any]]:
        return await self.get("odds", game=game_id)


DEFAULT_PRIORITY_LEAGUES = (
    "Champions League",
    "EHF European League",
    "Bundesliga",
    "Starligue",
    "Liga ASOBAL",
    "Herre Handbold Ligaen",
    "REMA 1000-ligaen",
    "Handbollsligan",
    "Extraliga",
)


def upcoming_days(days: int, now: datetime | None = None) -> list[date]:
    current = (now or datetime.now(timezone.utc)).date()
    return [current + timedelta(days=offset) for offset in range(max(1, days))]


def league_allowed(game: dict[str, Any], league_names: set[str]) -> bool:
    league = str((game.get("league") or {}).get("name", "")).strip().casefold()
    return not league_names or league in league_names


def normalize_game(game: dict[str, Any], odds_payload: list[dict[str, Any]]) -> dict[str, Any] | None:
    game_id = game.get("id")
    teams = game.get("teams") or {}
    home = str((teams.get("home") or {}).get("name", "")).strip()
    away = str((teams.get("away") or {}).get("name", "")).strip()
    if game_id in (None, "") or not home or not away:
        return None

    bookmakers: list[dict[str, Any]] = []
    for offer in odds_payload:
        for book in offer.get("bookmakers") or []:
            outcomes: list[dict[str, Any]] = []
            for bet in book.get("bets") or []:
                bet_name = str(bet.get("name", "")).casefold()
                if not any(token in bet_name for token in ("winner", "match result", "1x2")):
                    continue
                for value in bet.get("values") or []:
                    raw = str(value.get("value", "")).strip()
                    mapped = {"home": home, "1": home, "draw": "Draw", "x": "Draw",
                              "away": away, "2": away}.get(raw.casefold(), raw)
                    try:
                        price = float(value.get("odd", value.get("odds", 0)) or 0)
                    except (TypeError, ValueError):
                        continue
                    if mapped and price > 1.01:
                        outcomes.append({"name": mapped, "price": price})
            if outcomes:
                bookmakers.append({
                    "title": str(book.get("name", book.get("title", "API-Sports"))),
                    "markets": [{"key": "h2h", "outcomes": outcomes}],
                })

    league = game.get("league") or {}
    return {
        "id": f"apisports:{game_id}",
        "home_team": home,
        "away_team": away,
        "commence_time": str(game.get("date", "")),
        "league_key": f"api_sports_{league.get('id', 'unknown')}",
        "league_name": str(league.get("name", "Handball")),
        "bookmakers": bookmakers,
    }
