from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from core.config import Settings
from core.market import consensus_h2h, best_outlier_prices, dedupe_best_bets
from core.odds_api import fetch_odds
from core.monte_carlo import simulate_single_bet
from core.sport_quant import (
    bookmaker_grade,
    discover_active_sport_keys,
    elo_adjustment,
    filter_active_keys,
    init_sport_db,
    refresh_bookmaker_stats,
    sport_analytics_report,
    update_closing_lines,
)
from core.sport_settlement import settle_sport_bets
from core.staking import kelly_stake
from core.types import Bet, SportResult
from sports.base import SportModule
from core.meta_model import MetaFeatures, predict_probability
from core.adaptive_weights import (
    sport_weight,
    bookmaker_weight,
    league_weight,
)





LIGY: Dict[str, Dict[str, Any]] = {
    # === TOP EURÓPSKE LIGY ===
    "Premier League": {
        "csv": "E0",
        "api": "soccer_epl",
        "aliases": [],
        "ha": 0.35,
        "tier": "A",
    },
    "La Liga": {
        "csv": "SP1",
        "api": "soccer_spain_la_liga",
        "aliases": [],
        "ha": 0.38,
        "tier": "A",
    },
    "Bundesliga": {
        "csv": "D1",
        "api": "soccer_germany_bundesliga",
        "aliases": [],
        "ha": 0.40,
        "tier": "A",
    },
    "Serie A": {
        "csv": "I1",
        "api": "soccer_italy_serie_a",
        "aliases": [],
        "ha": 0.30,
        "tier": "A",
    },
    "Ligue 1": {
        "csv": "F1",
        "api": "soccer_france_ligue_one",
        "aliases": ["soccer_france_ligue_1"],
        "ha": 0.32,
        "tier": "A",
    },
    "Eredivisie": {
        "csv": "N1",
        "api": "soccer_netherlands_eredivisie",
        "aliases": [],
        "ha": 0.42,
        "tier": "A",
    },
    "Championship": {
        "csv": "E1",
        "api": "soccer_efl_champ",
        "aliases": [],
        "ha": 0.28,
        "tier": "A",
    },

    # === ANGLICKO & ŠKÓTSKO ===
    "League One": {
        "csv": "E2",
        "api": "soccer_england_league1",
        "aliases": ["soccer_england_league_one"],
        "ha": 0.30,
        "tier": "B",
    },
    "League Two": {
        "csv": "E3",
        "api": "soccer_england_league2",
        "aliases": ["soccer_england_league_two"],
        "ha": 0.30,
        "tier": "B",
    },
    "Scottish Premiership": {
        "csv": "SC0",
        "api": "soccer_spl",
        "aliases": ["soccer_scotland_premiership"],
        "ha": 0.32,
        "tier": "B",
    },
    "Scottish Championship": {
        "csv": "SC1",
        "api": "soccer_scotland_championship",
        "aliases": [],
        "ha": 0.30,
        "tier": "B",
    },

    # === OSTATNÁ EURÓPA ===
    "Primeira Liga": {
        "csv": "P1",
        "api": "soccer_portugal_primeira_liga",
        "aliases": [],
        "ha": 0.34,
        "tier": "A",
    },
    "Belgian Pro League": {
        "csv": "B1",
        "api": "soccer_belgium_first_div",
        "aliases": [],
        "ha": 0.33,
        "tier": "A",
    },
    "Turkish Super Lig": {
        "csv": "T1",
        "api": "soccer_turkey_super_league",
        "aliases": [],
        "ha": 0.36,
        "tier": "B",
    },
    "Greek Super League": {
        "csv": "G1",
        "api": "soccer_greece_super_league",
        "aliases": [],
        "ha": 0.35,
        "tier": "B",
    },
    "Austrian Bundesliga": {
        "csv": "AUT1",
        "api": "soccer_austria_bundesliga",
        "aliases": [],
        "ha": 0.34,
        "tier": "B",
    },
    "Swiss Super League": {
        "csv": "SWZ1",
        "api": "soccer_switzerland_superleague",
        "aliases": ["soccer_switzerland_super_league"],
        "ha": 0.32,
        "tier": "B",
    },
    "Danish Superliga": {
        "csv": "DNK1",
        "api": "soccer_denmark_superliga",
        "aliases": [],
        "ha": 0.34,
        "tier": "B",
    },
    "Eliteserien": {
        "csv": "NOR1",
        "api": "soccer_norway_eliteserien",
        "aliases": [],
        "ha": 0.39,
        "tier": "B",
    },
    "Allsvenskan": {
        "csv": "SWE1",
        "api": "soccer_sweden_allsvenskan",
        "aliases": [],
        "ha": 0.38,
        "tier": "B",
    },

    # === DRUHÉ EURÓPSKE LIGY ===
    "2. Bundesliga": {
        "csv": "D2",
        "api": "soccer_germany_bundesliga2",
        "aliases": ["soccer_germany_2bundesliga"],
        "ha": 0.32,
        "tier": "B",
    },
    "Segunda Division": {
        "csv": "SP2",
        "api": "soccer_spain_segunda_division",
        "aliases": [],
        "ha": 0.28,
        "tier": "B",
    },
    "Serie B": {
        "csv": "I2",
        "api": "soccer_italy_serie_b",
        "aliases": [],
        "ha": 0.26,
        "tier": "B",
    },
    "Ligue 2": {
        "csv": "F2",
        "api": "soccer_france_ligue_two",
        "aliases": ["soccer_france_ligue_2"],
        "ha": 0.25,
        "tier": "B",
    },

    # === SEVERNÁ & JUŽNÁ AMERIKA ===
    "MLS": {
        "csv": "USA1",
        "api": "soccer_usa_mls",
        "aliases": [],
        "ha": 0.42,
        "tier": "A",
    },
    "Brazil Serie A": {
        "csv": "BRA1",
        "api": "soccer_brazil_campeonato",
        "aliases": [],
        "ha": 0.28,
        "tier": "A",
    },
    "Argentina Primera Division": {
        "csv": "ARG1",
        "api": "soccer_argentina_primera_division",
        "aliases": [],
        "ha": 0.27,
        "tier": "A",
    },
    "Liga MX": {
        "csv": "MEX1",
        "api": "soccer_mexico_ligamx",
        "aliases": ["soccer_mexico_liga_mx"],
        "ha": 0.45,
        "tier": "A",
    },

    # === ÁZIA & OCEÁNIA ===
    "J-League": {
        "csv": "JPN1",
        "api": "soccer_japan_j_league",
        "aliases": [],
        "ha": 0.32,
        "tier": "B",
    },
    "A-League": {
        "csv": "AUS1",
        "api": "soccer_australia_aleague",
        "aliases": [],
        "ha": 0.35,
        "tier": "B",
    },

    # === MEDZINÁRODNÉ SÚŤAŽE ===
    "UEFA Champions League": {
        "csv": "UCL",
        "api": "soccer_uefa_champs_league",
        "aliases": [],
        "ha": 0.15,
        "tier": "A",
    },
    "UEFA Europa League": {
        "csv": "UEL",
        "api": "soccer_uefa_europa_league",
        "aliases": [],
        "ha": 0.15,
        "tier": "A",
    },
    "UEFA Conference League": {
        "csv": "UECL",
        "api": "soccer_uefa_europa_conference_league",
        "aliases": ["soccer_uefa_conference_league"],
        "ha": 0.15,
        "tier": "A",
    },
    "UEFA Nations League": {
        "csv": "UNL",
        "api": "soccer_uefa_nations_league",
        "aliases": [],
        "ha": 0.20,
        "tier": "A",
    },
    "FIFA World Cup": {
        "csv": "WC",
        "api": "soccer_fifa_world_cup",
        "aliases": [],
        "ha": 0.18,
        "tier": "A",
    },
    "Club World Cup": {
        "csv": "CWC",
        "api": "soccer_fifa_club_world_cup",
        "aliases": [],
        "ha": 0.15,
        "tier": "A",
    },

    # === KVALIFIKÁCIE ===
    "World Cup Qualifiers Europe": {
        "csv": "WCQEU",
        "api": "soccer_fifa_world_cup_qualifiers_europe",
        "aliases": [],
        "ha": 0.20,
        "tier": "A",
    },
    "World Cup Qualifiers South America": {
        "csv": "WCQSA",
        "api": "soccer_conmebol_world_cup_qualifiers",
        "aliases": ["soccer_fifa_world_cup_qualifiers_south_america"],
        "ha": 0.28,
        "tier": "A",
    },
}


def configured_football_keys() -> list[str]:
    """Return every preferred key and alias without duplicates."""
    keys: list[str] = []

    for config in LIGY.values():
        preferred = str(config.get("api", "")).strip()

        if preferred:
            keys.append(preferred)

        for alias in config.get("aliases", []):
            alias = str(alias).strip()

            if alias:
                keys.append(alias)

    return list(dict.fromkeys(keys))


def league_config_by_key(sport_key: str) -> tuple[str, Dict[str, Any]] | None:
    for league_name, config in LIGY.items():
        keys = {
            str(config.get("api", "")).strip(),
            *[
                str(alias).strip()
                for alias in config.get("aliases", [])
            ],
        }

        if sport_key in keys:
            return league_name, config

    return None


def _mc_probability(result: Any) -> float:
    """Read Monte Carlo probability across older/newer result objects."""
    value = getattr(result, "win_probability", None)
    if value is None:
        value = getattr(result, "simulated_win_probability", None)
    if value is None:
        raise AttributeError("Monte Carlo result has no win probability field")
    return max(0.01, min(0.99, float(value)))

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_hash(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class FootballModule(SportModule):
    name = "football"

    def _db_path(self, settings: Settings) -> Path:
        return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))

    def _connect(self, settings: Settings) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path(settings))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _save_snapshot(
        self,
        settings: Settings,
        sport_key: str,
        event_name: str,
        home: str,
        away: str,
        bookmakers: list[dict],
    ) -> int:
        if os.getenv("SNAPSHOT_ODDS", "1") != "1":
            return 0

        captured_at = now_utc()
        rows = []

        for bookmaker in bookmakers:
            book = str(bookmaker.get("title", ""))

            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    selection = str(outcome.get("name", ""))
                    odds = float(outcome.get("price", 0) or 0)

                    if odds <= 1.01:
                        continue

                    source_hash = make_hash(
                        captured_at,
                        self.name,
                        sport_key,
                        event_name,
                        book,
                        "h2h",
                        selection,
                        odds,
                    )

                    rows.append((
                        captured_at,
                        self.name,
                        sport_key,
                        event_name,
                        home,
                        away,
                        book,
                        "h2h",
                        selection,
                        odds,
                        source_hash,
                    ))

        if not rows:
            return 0

        with self._connect(settings) as conn:
            before = conn.total_changes
            conn.executemany("""
                INSERT OR IGNORE INTO sport_odds_snapshots
                (
                    captured_at, sport, league, event, home_team, away_team,
                    bookmaker, market, selection, odds, source_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

            return conn.total_changes - before

    def _save_bet(self, settings: Settings, bet: Bet) -> None:
        source_hash = make_hash(
            bet.sport,
            bet.league,
            bet.event,
            bet.market,
            bet.selection,
            bet.start_time,
        )

        with self._connect(settings) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO sport_bets
                (
                    sport, league, event, home_team, away_team, market,
                    selection, odds, prob_model, prob_market, prob_final,
                    edge, stake, bookmaker, start_time, score, source_hash,
                    result
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bet.sport,
                bet.league,
                bet.event,
                bet.event.split(" vs ")[0] if " vs " in bet.event else "",
                bet.event.split(" vs ")[1] if " vs " in bet.event else "",
                bet.market,
                bet.selection,
                bet.odds,
                bet.prob_model,
                bet.prob_market,
                bet.prob_final,
                bet.edge,
                bet.stake,
                bet.bookmaker,
                bet.start_time,
                bet.score,
                source_hash,
                "OPEN",
            ))

    def _audit(
        self,
        settings: Settings,
        sport_key: str,
        event_name: str,
        selection: str,
        bookmaker: str,
        odds: float,
        prob_market: float | None,
        edge: float | None,
        decision: str,
        reason: str,
    ) -> None:
        with self._connect(settings) as conn:
            conn.execute("""
                INSERT INTO sport_decision_audit
                (
                    sport, league, event, selection, bookmaker,
                    odds, prob_market, edge, decision, reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.name,
                sport_key,
                event_name,
                selection,
                bookmaker,
                odds,
                prob_market,
                edge,
                decision,
                reason,
            ))

    def _football_home_adjustment(
        self,
        sport_key: str,
        home: str,
        away: str,
        selection: str,
    ) -> float:
        if os.getenv("FOOTBALL_HOME_ADV_ENABLED", "1") != "1":
            return 0.0

        league_info = league_config_by_key(sport_key)
        base_ha = 0.35

        if league_info is not None:
            _, config = league_info
            base_ha = float(config.get("ha", base_ha))

        # Hodnoty ha v LIGY sú liga-level intenzita, nie priamo
        # percentuálne body pravdepodobnosti. Scale 0.05 znamená,
        # že ha=0.35 vytvorí úpravu približne +1.75 p. b.
        scale = float(os.getenv("FOOTBALL_HA_PROB_SCALE", "0.05"))
        probability_adjustment = max(
            0.0,
            min(0.04, base_ha * scale),
        )

        if selection == home:
            return probability_adjustment

        if selection == away:
            return -probability_adjustment

        return 0.0

    async def scan(self, settings: Settings) -> SportResult:
        init_sport_db(settings)

        default_keys = configured_football_keys()

        env_keys = [
            item.strip()
            for item in os.getenv("FOOTBALL_SPORT_KEYS", "").split(",")
            if item.strip()
        ]

        requested_keys = env_keys or default_keys
        clean_sport_keys = list(requested_keys)

        if os.getenv("SPORT_KEY_AUTO_DISCOVERY", "1") == "1":
            active_keys = await discover_active_sport_keys(
                settings.odds_api_key,
                ["Soccer"],
            )

            active_key_set = {
                str(item).strip()
                for item in active_keys
                if str(item).strip()
            }

            clean_sport_keys = [
                key
                for key in requested_keys
                if key in active_key_set
            ]

        clean_sport_keys = list(dict.fromkeys(clean_sport_keys))

        settled = await settle_sport_bets(
            settings=settings,
            sport=self.name,
            sport_keys=clean_sport_keys,
        )

        updated_clv = update_closing_lines(settings, self.name)
        refresh_bookmaker_stats(settings, self.name)

        min_books = int(os.getenv("MIN_FOOTBALL_BOOKMAKERS", "3"))
        top_n = int(os.getenv("TOP_N_REPORT", "8"))
        grade_min_samples = int(os.getenv("FOOTBALL_BOOKMAKER_GRADE_MIN_SAMPLES", "20"))

        bets: list[Bet] = []
        snapshots_saved = 0
        blocked = 0
        scanned_events = 0

        for sport_key in clean_sport_keys:
            data = await fetch_odds(
                settings.odds_api_key,
                sport_key,
                markets="h2h",
            )

            if not data:
                continue

            for event in data:
                league_info = league_config_by_key(sport_key)
                league = (
                    league_info[0]
                    if league_info is not None
                    else sport_key
                )
                home = str(event.get("home_team", ""))
                away = str(event.get("away_team", ""))
                start = str(event.get("commence_time", ""))
                event_name = f"{home} vs {away}"
                bookmakers = event.get("bookmakers", [])

                scanned_events += 1
                snapshots_saved += self._save_snapshot(
                    settings=settings,
                    sport_key=sport_key,
                    event_name=event_name,
                    home=home,
                    away=away,
                    bookmakers=bookmakers,
                )

                consensus = consensus_h2h(bookmakers, min_books=min_books)

                if not consensus:
                    blocked += 1
                    self._audit(
                        settings,
                        sport_key,
                        event_name,
                        "",
                        "",
                        0,
                        None,
                        None,
                        "BLOCK",
                        "no market consensus",
                    )
                    continue

                for bookmaker, selection, odds in best_outlier_prices(bookmakers):
                    prob_market = consensus.get(selection)

                    if not prob_market:
                        blocked += 1
                        self._audit(
                            settings,
                            sport_key,
                            event_name,
                            selection,
                            bookmaker,
                            odds,
                            None,
                            None,
                            "BLOCK",
                            "selection missing in consensus",
                        )
                        continue

                    grade = bookmaker_grade(
                        settings,
                        self.name,
                        bookmaker,
                        min_samples=grade_min_samples,
                    )

                    elo_adj = elo_adjustment(
                        settings,
                        self.name,
                        home,
                        away,
                        selection,
                    )

                    home_adj = self._football_home_adjustment(
                        sport_key=sport_key,
                        home=home,
                        away=away,
                        selection=selection,
                    )

                    fallback_probability = max(
                        0.01,
                        min(0.99, prob_market + elo_adj + home_adj),
                    )

                    current_sport_weight = sport_weight(self.name)
                    current_bookmaker_weight = bookmaker_weight(bookmaker)
                    current_league_weight = league_weight(league)

                    mc_preview = simulate_single_bet(
                        probability=fallback_probability,
                        odds=odds,
                    )

                    probability_source = "FALLBACK"
                    probability_reason = "original football probability"

                    try:
                        features = MetaFeatures(
                            market_probability=prob_market,
                            elo_adjustment=elo_adj,
                            form_adjustment=0.0,
                            clv_adjustment=0.0,
                            bookmaker_grade=grade,
                            sport_weight=current_sport_weight,
                            league_weight=current_league_weight,
                            confidence=50.0,
                            monte_carlo_probability=_mc_probability(mc_preview),
                        )
                        prob_final = max(0.01, min(0.99, predict_probability(features)))
                        probability_source = "META_MODEL"
                        probability_reason = f"meta_probability={prob_final:.4f}"
                    except Exception as exc:
                        prob_final = fallback_probability
                        probability_reason = f"{type(exc).__name__}: {exc}"

                    edge = prob_final * odds - 1.0
                    adjusted_edge = (
                        edge
                        * grade
                        * current_sport_weight
                        * current_bookmaker_weight
                        * current_league_weight
                    )

                    if edge < settings.min_edge:
                        blocked += 1
                        self._audit(
                            settings,
                            sport_key,
                            event_name,
                            selection,
                            bookmaker,
                            odds,
                            prob_market,
                            edge,
                            "BLOCK",
                            f"edge below minimum; probability_source={probability_source}; probability_reason={probability_reason}",
                        )
                        continue

                    if edge > settings.max_edge:
                        blocked += 1
                        self._audit(
                            settings,
                            sport_key,
                            event_name,
                            selection,
                            bookmaker,
                            odds,
                            prob_market,
                            edge,
                            "BLOCK",
                            f"edge above max guard; probability_source={probability_source}; probability_reason={probability_reason}",
                        )
                        continue

                    if odds > settings.max_odds:
                        blocked += 1
                        self._audit(
                            settings,
                            sport_key,
                            event_name,
                            selection,
                            bookmaker,
                            odds,
                            prob_market,
                            edge,
                            "BLOCK",
                            f"odds above max odds; probability_source={probability_source}; probability_reason={probability_reason}",
                        )
                        continue

                    stake = kelly_stake(prob_final, odds, settings)
                    stake = round(stake * grade, 2)

                    if stake <= 0:
                        blocked += 1
                        self._audit(
                            settings,
                            sport_key,
                            event_name,
                            selection,
                            bookmaker,
                            odds,
                            prob_market,
                            edge,
                            "BLOCK",
                            f"stake <= 0; probability_source={probability_source}; probability_reason={probability_reason}",
                        )
                        continue

                    bet = Bet(
                        sport=self.name,
                        league=league,
                        event=event_name,
                        market="h2h",
                        selection=selection,
                        odds=odds,
                        prob_model=prob_market,
                        prob_market=prob_market,
                        prob_final=prob_final,
                        edge=edge,
                        stake=stake,
                        bookmaker=bookmaker,
                        start_time=start,
                        score=adjusted_edge * 100,
                    )

                    bets.append(bet)
                    self._save_bet(settings, bet)

                    self._audit(
                        settings,
                        sport_key,
                        event_name,
                        selection,
                        bookmaker,
                        odds,
                        prob_market,
                        edge,
                        "PASS",
                        f"league={league}; bookmaker grade {grade:.2f}, "
                        f"elo_adj={elo_adj:.3f}, home_adj={home_adj:.3f}, "
                        f"probability_source={probability_source}, "
                        f"probability_reason={probability_reason}",
                    )

        bets = dedupe_best_bets(bets)
        analytics = sport_analytics_report(settings, self.name)

        return SportResult(
            sport=self.name,
            mode="scan",
            bets=bets[:top_n],
            message=(
                "Football v12 Final: league registry/meta-model/fallback/"
                "CLV/ELO/Monte Carlo model. "
                f"Settled: {settled}. "
                f"CLV updated: {updated_clv}. "
                f"Events scanned: {scanned_events}. "
                f"Snapshots saved: {snapshots_saved}. "
                f"Blocked: {blocked}. "
                f"Stored candidates: {len(bets)}.\n"
                f"{analytics}"
            ),
        )
