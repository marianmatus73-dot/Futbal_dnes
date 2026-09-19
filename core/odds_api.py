from __future__ import annotations

import aiohttp
import logging

log = logging.getLogger("odds-api")


async def fetch_odds(
    api_key: str,
    sport_key: str,
    markets: str = "h2h",
    regions: str = "eu",
) -> list[dict]:
    if not api_key:
        log.warning("Missing ODDS_API_KEY.")
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.warning("Odds API %s for %s: %s", resp.status, sport_key, body[:300])
                    return []
                return await resp.json()
        except Exception as e:
            log.warning("Odds API error for %s: %s", sport_key, e)
            return []


async def fetch_event_odds(
    api_key: str,
    sport_key: str,
    event_id: str,
    markets: str = "double_chance",
    regions: str = "eu",
) -> dict:
    """Fetch an additional market, which the provider exposes per event only."""
    if not api_key or not event_id:
        return {}
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport_key}"
        f"/events/{event_id}/odds/"
    )
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=12) as resp:
                if resp.status != 200:
                    log.warning(
                        "Odds API event market %s for %s/%s: %s",
                        resp.status, sport_key, event_id, (await resp.text())[:200],
                    )
                    return {}
                payload = await resp.json()
                return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        log.warning("Odds API event market error for %s/%s: %s", sport_key, event_id, exc)
        return {}
