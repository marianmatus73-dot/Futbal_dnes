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
