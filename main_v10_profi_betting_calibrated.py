#!/usr/bin/env python3
"""
PRODUCTION FOOTBALL BETTING MODEL v10 PROFI
-------------------------------------------------
Vychadza z tvojej v6.0 verzie a pridava:
- robustnejsiu konfiguraciu cez .env
- SQLite schema v2 + unikaty proti duplicitnym tipom
- migraciu zo starsieho bets.db / historia_tipov.csv
- automaticke vyhodnocovanie vysledkov
- fuzzy matching nazvov timov
- Poisson model s recent formou a home advantage
- volitelny LightGBM AI filter, ked je dost historie
- value bet scoring, Kelly staking, capy a bankroll ochranu
- CSV export reportu + email report
- lepsie logovanie a odolnost pri chybach API

POZOR: Toto nie je garancia zisku. Pouzivaj zodpovedne a s vlastnym risk manazmentom.
"""

from __future__ import annotations

import argparse
import asyncio
import aiohttp
import csv
import difflib
import hashlib
import io
import json
import logging
import os
import sqlite3
import smtplib
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import poisson

try:
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
except Exception:  # LightGBM/sklearn mozu byt volitelne
    LGBMClassifier = None
    CalibratedClassifierCV = None
    StratifiedKFold = None


# =============================== CONFIG ===============================

load_dotenv()

APP_VERSION = "10.2-profi-calibrated"
DB_SCHEMA_VERSION = 3

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / os.getenv("CACHE_DIR", "cache")
EXPORT_DIR = ROOT / os.getenv("EXPORT_DIR", "exports")
CACHE_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

DB_FILE = ROOT / os.getenv("DB_FILE", "bets.db")
HISTORY_CSV = ROOT / os.getenv("HISTORY_CSV", "historia_tipov.csv")

API_ODDS_KEY = os.getenv("ODDS_API_KEY", "").strip()
GMAIL_USER = os.getenv("GMAIL_USER", "").strip()
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD", "").strip()
GMAIL_RECEIVER = os.getenv("GMAIL_RECEIVER", GMAIL_USER).strip()

def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


BANK = env_float("AKTUALNY_BANK", 1000.0)
KELLY_FRAC = env_float("KELLY_FRAC", 0.03)
MAX_STAKE_PCT = env_float("MAX_STAKE_PCT", 0.015)
MIN_STAKE = env_float("MIN_STAKE", 1.0)
MIN_EDGE = env_float("MIN_EDGE", 0.06)
MAX_EDGE = env_float("MAX_EDGE", 0.15)
MIN_AI_PROB = env_float("MIN_AI_PROB", 0.54)
LOOKAHEAD_HOURS = env_int("LOOKAHEAD_HOURS", 72)
CACHE_TTL_SECONDS = env_int("CACHE_TTL_SECONDS", 3600)
RECENT_MATCHES = env_int("RECENT_MATCHES", 8)
MAX_GOALS_GRID = env_int("MAX_GOALS_GRID", 10)
HTTP_TIMEOUT = env_int("HTTP_TIMEOUT", 30)
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Bratislava"))
SEASON_CODE = os.getenv("FOOTBALL_DATA_SEASON", "").strip()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("betting-v10")


LIGY: Dict[str, Dict[str, Any]] = {
    "Premier League": {"csv": "E0", "api": "soccer_epl", "ha": 0.35},
    "La Liga": {"csv": "SP1", "api": "soccer_spain_la_liga", "ha": 0.38},
    "Bundesliga": {"csv": "D1", "api": "soccer_germany_bundesliga", "ha": 0.40},
    "Serie A": {"csv": "I1", "api": "soccer_italy_serie_a", "ha": 0.30},
    "Ligue 1": {"csv": "F1", "api": "soccer_france_ligue_one", "ha": 0.32},
    "Eredivisie": {"csv": "N1", "api": "soccer_netherlands_eredivisie", "ha": 0.42},
    "Championship": {"csv": "E1", "api": "soccer_efl_champ", "ha": 0.28},
}

ACTIVE_LEAGUES: Optional[set[str]] = None
DRY_RUN = False
SETTLE_ONLY = False
NO_EMAIL = False
BACKTEST_ONLY = False
ANALYTICS_ONLY = False
ANALYTICS_DAYS = env_int("ANALYTICS_DAYS", 365)
BACKTEST_DAYS = env_int("BACKTEST_DAYS", 120)
MIN_PROB = env_float("MIN_PROB", 0.18)
MAX_ODDS = env_float("MAX_ODDS", 7.50)
TOP_N_REPORT = env_int("TOP_N_REPORT", 8)
MAX_BETS_PER_DAY = env_int("MAX_BETS_PER_DAY", 8)
MAX_DAILY_EXPOSURE_PCT = env_float("MAX_DAILY_EXPOSURE_PCT", 0.05)
MAX_MATCH_EXPOSURE_PCT = env_float("MAX_MATCH_EXPOSURE_PCT", 0.015)
MAX_LEAGUE_EXPOSURE_PCT = env_float("MAX_LEAGUE_EXPOSURE_PCT", 0.035)
REQUIRE_MARKET_AGREEMENT = env_int("REQUIRE_MARKET_AGREEMENT", 0)  # 1 = at least 2 bookmakers for same pick
MIN_BOOKMAKERS_AGREE = env_int("MIN_BOOKMAKERS_AGREE", 2)
DIXON_COLES_RHO = env_float("DIXON_COLES_RHO", -0.08)
MARKET_BLEND_WEIGHT = env_float("MARKET_BLEND_WEIGHT", 0.55)
MIN_MARKET_EDGE = env_float("MIN_MARKET_EDGE", 0.035)  # edge vs no-vig consensus, model sanity guard
SNAPSHOT_ODDS = env_int("SNAPSHOT_ODDS", 1)  # store raw odds snapshots for CLV tracking
CLV_LOOKBACK_HOURS = env_int("CLV_LOOKBACK_HOURS", 12)
PROB_SHRINK = env_float("PROB_SHRINK", 0.72)
MAX_MODEL_MARKET_GAP = env_float("MAX_MODEL_MARKET_GAP", 0.24)
MIN_CONSENSUS_BOOKMAKERS = env_int("MIN_CONSENSUS_BOOKMAKERS", 2)




@dataclass(frozen=True)
class BetCandidate:
    datum_iso: str
    datum_display: str
    league: str
    zapas: str
    home_team: str
    away_team: str
    tip: str
    market: str
    kurz: float
    prob_model: float
    prob_market: Optional[float]
    prob_final: float
    edge: float
    market_edge: Optional[float]
    lh: float
    la: float
    vklad: float
    bookmaker: str
    ai_prob: Optional[float]
    score: float
    source_hash: str

    @property
    def implied_prob(self) -> float:
        return round(1.0 / self.kurz, 5) if self.kurz > 0 else 0.0

    @property
    def fair_odds(self) -> float:
        return round(1.0 / self.prob_final, 3) if self.prob_final > 0 else 0.0

    @property
    def ev_eur(self) -> float:
        return round(self.vklad * self.edge, 2)

    @property
    def risk_level(self) -> str:
        if self.kurz >= 4.5 or self.edge >= 0.18:
            return "HIGH"
        if self.kurz >= 2.8 or self.edge >= 0.10:
            return "MEDIUM"
        return "LOW"

    def as_db_tuple(self) -> Tuple[Any, ...]:
        return (
            self.datum_iso,
            self.datum_display,
            self.league,
            self.zapas,
            self.home_team,
            self.away_team,
            self.tip,
            self.market,
            self.kurz,
            self.prob_model,
            self.prob_market,
            self.prob_final,
            self.edge,
            self.market_edge,
            self.lh,
            self.la,
            self.vklad,
            self.bookmaker,
            self.ai_prob,
            self.score,
            self.source_hash,
            "",
            APP_VERSION,
        )


# =============================== HELPERS ===============================

def parse_utc(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def football_data_season_code(ref: Optional[datetime] = None) -> str:
    """Return football-data.co.uk season code, e.g. 2526.

    European football seasons usually switch in July. An override is supported
    through FOOTBALL_DATA_SEASON for reproducible backtests or emergency fixes.
    """
    if SEASON_CODE:
        return SEASON_CODE
    ref = ref or now_utc()
    year = ref.year
    start = year if ref.month >= 7 else year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def normalize_name(name: str) -> str:
    return " ".join(str(name).lower().replace("fc", "").replace("afc", "").split())


def match_team(name: str, teams: Iterable[str], cutoff: float = 0.58) -> Optional[str]:
    teams_list = list(teams)
    if name in teams_list:
        return name
    lookup = {normalize_name(t): t for t in teams_list}
    norm = normalize_name(name)
    if norm in lookup:
        return lookup[norm]
    matches = difflib.get_close_matches(norm, list(lookup.keys()), n=1, cutoff=cutoff)
    return lookup[matches[0]] if matches else None


def make_hash(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def kelly_stake(prob: float, odds: float) -> float:
    if odds <= 1 or prob <= 0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    kelly = ((b * prob) - q) / b
    stake_pct = min(max(0.0, kelly * KELLY_FRAC), MAX_STAKE_PCT)
    return round(stake_pct * BANK, 2)


def value_score(prob: float, odds: float, edge: float, lh: float, la: float) -> float:
    """Conservative score: value + probability - fragility penalties.

    Higher odds and extremely high model edge are penalized because they are
    often produced by stale odds, team-name mismatches, or model misspecification.
    """
    odds_penalty = max(0.0, odds - 3.0) * 2.0
    edge_penalty = max(0.0, edge - 0.10) * 45
    goal_uncertainty = abs(lh - la) * 0.15
    probability_bonus = min(prob, 0.72) * 10
    return round(edge * 100 + probability_bonus - odds_penalty - edge_penalty + goal_uncertainty, 4)


def is_reasonable_candidate(prob: float, odds: float, edge: float) -> bool:
    if odds <= 1.01 or odds > MAX_ODDS:
        return False
    if prob < MIN_PROB:
        return False
    if edge < MIN_EDGE:
        return False
    # Keep MAX_EDGE as a data-quality guardrail. Unrealistic edge is more often
    # bad data than a free lunch.
    if edge > MAX_EDGE:
        return False
    return True


# =============================== HTTP / DATA ===============================

async def fetch_text(session: aiohttp.ClientSession, url: str, params: Optional[dict] = None) -> Optional[str]:
    try:
        async with session.get(url, params=params, timeout=HTTP_TIMEOUT) as resp:
            if resp.status != 200:
                log.warning("HTTP %s pri %s", resp.status, url)
                return None
            return await resp.text()
    except Exception as e:
        log.warning("HTTP chyba pri %s: %s", url, e)
        return None


async def fetch_csv(session: aiohttp.ClientSession, url: str) -> Optional[pd.DataFrame]:
    cache_file = CACHE_DIR / f"{Path(url).stem}.csv"
    if cache_file.exists():
        age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            try:
                return pd.read_csv(cache_file)
            except Exception:
                pass

    txt = await fetch_text(session, url)
    if not txt:
        return None
    try:
        df = pd.read_csv(io.StringIO(txt))
        df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        log.warning("CSV parse chyba %s: %s", url, e)
        return None


async def fetch_odds(session: aiohttp.ClientSession, sport_key: str) -> List[dict]:
    if not API_ODDS_KEY:
        log.warning("Chyba ODDS_API_KEY v .env - odds API preskakujem.")
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": API_ODDS_KEY,
        "regions": os.getenv("ODDS_REGIONS", "eu"),
        "markets": os.getenv("ODDS_MARKETS", "h2h,totals"),
        "oddsFormat": "decimal",
    }
    try:
        async with session.get(url, params=params, timeout=HTTP_TIMEOUT) as resp:
            if resp.status != 200:
                body = await resp.text()
                log.warning("Odds API %s pre %s: %s", resp.status, sport_key, body[:200])
                return []
            return await resp.json()
    except Exception as e:
        log.warning("Odds API chyba %s: %s", sport_key, e)
        return []


# =============================== DB ===============================

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datum_iso TEXT,
                datum TEXT,
                league TEXT,
                zapas TEXT,
                home_team TEXT,
                away_team TEXT,
                tip TEXT,
                market TEXT,
                kurz REAL,
                prob_model REAL,
                prob_market REAL,
                prob_final REAL,
                edge REAL,
                market_edge REAL,
                lh REAL,
                la REAL,
                vklad REAL,
                bookmaker TEXT,
                ai_prob REAL,
                score REAL,
                source_hash TEXT UNIQUE,
                result TEXT,
                app_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                settled_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_result ON bets(result)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_datum_iso ON bets(datum_iso)")
        ensure_schema_v3(conn)
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)", (str(DB_SCHEMA_VERSION),))
        migrate_old_csv(conn)



def ensure_schema_v3(conn: sqlite3.Connection) -> None:
    """Add v3 columns/tables without breaking existing bets.db files."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(bets)").fetchall()}
    for col, ddl in {
        "prob_market": "ALTER TABLE bets ADD COLUMN prob_market REAL",
        "prob_final": "ALTER TABLE bets ADD COLUMN prob_final REAL",
        "market_edge": "ALTER TABLE bets ADD COLUMN market_edge REAL",
        "closing_kurz": "ALTER TABLE bets ADD COLUMN closing_kurz REAL",
        "clv_pct": "ALTER TABLE bets ADD COLUMN clv_pct REAL",
    }.items():
        if col not in existing:
            conn.execute(ddl)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            commence_time TEXT,
            league TEXT,
            home_team TEXT,
            away_team TEXT,
            bookmaker TEXT,
            market TEXT,
            tip TEXT,
            point REAL,
            price REAL,
            source_hash TEXT UNIQUE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_odds_snapshots_match ON odds_snapshots(commence_time, league, home_team, away_team, market, tip)")


def migrate_old_csv(conn: sqlite3.Connection) -> None:
    if not HISTORY_CSV.exists():
        return
    count = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
    if count > 0:
        return
    log.info("Migrujem historiu z %s", HISTORY_CSV.name)
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception as e:
        log.warning("CSV migracia zlyhala: %s", e)
        return
    for _, r in df.iterrows():
        datum = str(r.get("Datum", ""))
        zapas = str(r.get("Zápas", r.get("Zapas", "")))
        tip = str(r.get("Tip", ""))
        kurz = safe_float(r.get("Kurz", 0))
        edge = safe_float(r.get("Edge", 0))
        lh = safe_float(r.get("lh", 1.5), 1.5)
        la = safe_float(r.get("la", 1.5), 1.5)
        vklad = safe_float(r.get("Vklad", 0))
        result = str(r.get("Vysledok", r.get("result", "")))
        source_hash = make_hash(datum, zapas, tip, kurz, "migration")
        conn.execute(
            """
            INSERT OR IGNORE INTO bets
            (datum_iso, datum, league, zapas, home_team, away_team, tip, market, kurz,
             prob_model, edge, lh, la, vklad, bookmaker, ai_prob, score, source_hash, result, app_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (datum, datum, "", zapas, "", "", tip, "", kurz, None, edge, lh, la, vklad, "", None, edge, source_hash, result, "migration"),
        )


def insert_bets(bets: Sequence[BetCandidate]) -> int:
    if not bets:
        return 0
    sql = """
        INSERT OR IGNORE INTO bets
        (datum_iso, datum, league, zapas, home_team, away_team, tip, market, kurz,
         prob_model, prob_market, prob_final, edge, market_edge, lh, la, vklad, bookmaker, ai_prob, score, source_hash, result, app_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with db_connect() as conn:
        before = conn.total_changes
        conn.executemany(sql, [b.as_db_tuple() for b in bets])
        return conn.total_changes - before


def save_odds_snapshots(league: str, odds_data: Sequence[dict]) -> int:
    """Persist raw odds observations so later runs can estimate CLV.

    This is intentionally raw and bookmaker-level. It avoids lookahead bias in
    future diagnostics because the timestamp is captured before settlement.
    """
    if not SNAPSHOT_ODDS or not odds_data:
        return 0
    captured_at = now_utc().isoformat()
    rows = []
    for match in odds_data:
        commence = str(match.get("commence_time", ""))
        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        for bookmaker in match.get("bookmakers", []):
            bk = str(bookmaker.get("title", ""))
            for market in bookmaker.get("markets", []):
                market_key = str(market.get("key", ""))
                for outcome in market.get("outcomes", []):
                    label = outcome_label(market_key, str(outcome.get("name", "")), home, away, outcome.get("point"))
                    if not label:
                        continue
                    tip, market_name = label
                    price = safe_float(outcome.get("price", 0))
                    if price <= 1.01:
                        continue
                    point = outcome.get("point")
                    point_val = safe_float(point, 0.0) if point is not None else None
                    h = make_hash(captured_at, commence, league, home, away, bk, market_name, tip, point_val, price)
                    rows.append((captured_at, commence, league, home, away, bk, market_name, tip, point_val, price, h))
    if not rows:
        return 0
    with db_connect() as conn:
        before = conn.total_changes
        conn.executemany("""
            INSERT OR IGNORE INTO odds_snapshots
            (captured_at, commence_time, league, home_team, away_team, bookmaker, market, tip, point, price, source_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        return conn.total_changes - before



def save_match_odds_snapshot(league: str, match: dict, home_norm: str, away_norm: str, start_iso: str) -> int:
    """Persist odds for one normalized fixture after successful team matching."""
    if not SNAPSHOT_ODDS:
        return 0
    captured_at = now_utc().isoformat()
    home_api = str(match.get("home_team", ""))
    away_api = str(match.get("away_team", ""))
    rows = []
    for bookmaker in match.get("bookmakers", []):
        bk = str(bookmaker.get("title", ""))
        for market in bookmaker.get("markets", []):
            market_key = str(market.get("key", ""))
            for outcome in market.get("outcomes", []):
                label = outcome_label(market_key, str(outcome.get("name", "")), home_api, away_api, outcome.get("point"))
                if not label:
                    continue
                tip, market_name = label
                price = safe_float(outcome.get("price", 0))
                if price <= 1.01:
                    continue
                point = outcome.get("point")
                point_val = safe_float(point, 0.0) if point is not None else None
                h = make_hash(captured_at, start_iso, league, home_norm, away_norm, bk, market_name, tip, point_val, price)
                rows.append((captured_at, start_iso, league, home_norm, away_norm, bk, market_name, tip, point_val, price, h))
    if not rows:
        return 0
    with db_connect() as conn:
        before = conn.total_changes
        conn.executemany("""
            INSERT OR IGNORE INTO odds_snapshots
            (captured_at, commence_time, league, home_team, away_team, bookmaker, market, tip, point, price, source_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        return conn.total_changes - before


def update_clv_metrics() -> int:
    """Estimate closing-line value from latest stored snapshot before kickoff."""
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT id, datum_iso, league, home_team, away_team, tip, market, kurz, bookmaker
            FROM bets
            WHERE (closing_kurz IS NULL OR clv_pct IS NULL)
              AND datum_iso IS NOT NULL AND datum_iso != ''
        """).fetchall()
        updates = []
        for bet_id, datum_iso, league, home, away, tip, market, taken, bookmaker in rows:
            taken = safe_float(taken, 0.0)
            if taken <= 1.01:
                continue
            kickoff = pd.to_datetime(datum_iso, errors="coerce", utc=True, dayfirst=True)
            if pd.isna(kickoff):
                continue
            lower = (kickoff - pd.Timedelta(hours=CLV_LOOKBACK_HOURS)).isoformat()
            upper = kickoff.isoformat()
            # Prefer the same bookmaker's last quote; fall back to market average.
            q = conn.execute("""
                SELECT price FROM odds_snapshots
                WHERE league=? AND home_team=? AND away_team=? AND tip=? AND market=?
                  AND bookmaker=? AND captured_at BETWEEN ? AND ?
                ORDER BY captured_at DESC LIMIT 1
            """, (league, home, away, tip, market, bookmaker, lower, upper)).fetchone()
            closing = safe_float(q[0], 0.0) if q else 0.0
            if closing <= 1.01:
                q = conn.execute("""
                    SELECT AVG(price) FROM (
                        SELECT bookmaker, price, MAX(captured_at)
                        FROM odds_snapshots
                        WHERE league=? AND home_team=? AND away_team=? AND tip=? AND market=?
                          AND captured_at BETWEEN ? AND ?
                        GROUP BY bookmaker
                    )
                """, (league, home, away, tip, market, lower, upper)).fetchone()
                closing = safe_float(q[0], 0.0) if q else 0.0
            if closing <= 1.01:
                continue
            clv_pct = (taken / closing) - 1.0
            updates.append((round(closing, 4), round(clv_pct, 5), bet_id))
        if updates:
            conn.executemany("UPDATE bets SET closing_kurz=?, clv_pct=? WHERE id=?", updates)
        return len(updates)


# =============================== SETTLEMENT ===============================

async def settle_results(session: aiohttp.ClientSession) -> int:
    """Settle open bets using football-data results, matched by teams AND date.

    The previous implementation matched only by teams, which can settle the
    wrong fixture when the same teams have met more than once in a season.
    This version uses the stored kickoff date and accepts only matches in a
    small date window around that kickoff.
    """
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT id, datum_iso, zapas, home_team, away_team, tip, league
            FROM bets
            WHERE result IS NULL OR result = ''
        """).fetchall()

    if not rows:
        return 0

    rows_by_league: Dict[str, List[Tuple[Any, ...]]] = {}
    unknown_league_rows: List[Tuple[Any, ...]] = []
    for row in rows:
        league = str(row[6] or "")
        if league in LIGY:
            rows_by_league.setdefault(league, []).append(row)
        else:
            unknown_league_rows.append(row)

    settled = 0
    for league_name, cfg in LIGY.items():
        candidate_rows = list(rows_by_league.get(league_name, []))
        # Migrated/legacy rows may not have a league stored; keep a fallback.
        candidate_rows.extend(unknown_league_rows)
        if not candidate_rows:
            continue

        url = f"https://www.football-data.co.uk/mmz4281/{football_data_season_code()}/{cfg['csv']}.csv"
        df = await fetch_csv(session, url)
        if df is None or df.empty:
            continue
        required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Date"]
        if any(col not in df.columns for col in required):
            log.warning("%s: CSV nema potrebne stlpce pre settlement.", league_name)
            continue

        df = df.dropna(subset=required).copy()
        if df.empty:
            continue
        df["match_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
        df = df.dropna(subset=["match_date"])
        csv_teams = set(df["HomeTeam"].astype(str)) | set(df["AwayTeam"].astype(str))

        updates: List[Tuple[str, str, int]] = []
        for row_id, datum_iso, zapas, home, away, tip, _league in candidate_rows:
            h = home or (zapas.split(" vs ")[0] if " vs " in zapas else "")
            a = away or (zapas.split(" vs ")[1] if " vs " in zapas else "")
            csv_h = match_team(h, csv_teams)
            csv_a = match_team(a, csv_teams)
            if not csv_h or not csv_a:
                continue

            kickoff_date: Optional[datetime.date] = None
            if datum_iso:
                parsed_dt = pd.to_datetime(datum_iso, errors="coerce", utc=True, dayfirst=True)
                if not pd.isna(parsed_dt):
                    kickoff_date = parsed_dt.date()

            m = df[(df["HomeTeam"] == csv_h) & (df["AwayTeam"] == csv_a)].copy()
            if m.empty:
                continue
            if kickoff_date is not None:
                # Allow one day of tolerance for timezone/date formatting differences.
                m["date_diff"] = m["match_date"].apply(lambda d: abs((d - kickoff_date).days))
                m = m[m["date_diff"] <= 1].sort_values("date_diff")
            if m.empty:
                continue

            r = m.iloc[0]
            fthg, ftag, ftr = int(r["FTHG"]), int(r["FTAG"]), str(r["FTR"])
            won = (
                (tip == "1" and ftr == "H")
                or (tip == "X" and ftr == "D")
                or (tip == "2" and ftr == "A")
                or (tip == "Over 2.5" and (fthg + ftag) >= 3)
                or (tip == "Under 2.5" and (fthg + ftag) <= 2)
            )
            updates.append(("V" if won else "P", now_utc().isoformat(), row_id))

        if updates:
            with db_connect() as conn:
                conn.executemany("UPDATE bets SET result=?, settled_at=? WHERE id=?", updates)
            settled += len(updates)
    if settled:
        log.info("Vyhodnotenych tipov: %s", settled)
    return settled

# =============================== MODELS ===============================

def dixon_coles_adjustment(i: int, j: int, lh: float, la: float, rho: float) -> float:
    """Low-score Dixon-Coles correction factor."""
    if i == 0 and j == 0:
        return max(0.01, 1.0 - (lh * la * rho))
    if i == 0 and j == 1:
        return max(0.01, 1.0 + (lh * rho))
    if i == 1 and j == 0:
        return max(0.01, 1.0 + (la * rho))
    if i == 1 and j == 1:
        return max(0.01, 1.0 - rho)
    return 1.0


def poisson_probs(lh: float, la: float, rho: Optional[float] = None) -> Dict[str, float]:
    """Poisson 1X2 and totals probabilities, optionally Dixon-Coles adjusted."""
    goals = np.arange(MAX_GOALS_GRID)
    matrix = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    if rho is not None and abs(rho) > 1e-12:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] *= dixon_coles_adjustment(i, j, lh, la, rho)
    s = matrix.sum()
    if s > 0:
        matrix = matrix / s
    total_goals = np.add.outer(goals, goals)
    return {
        "1": float(np.tril(matrix, -1).sum()),
        "X": float(np.diag(matrix).sum()),
        "2": float(np.triu(matrix, 1).sum()),
        "Over 2.5": float(matrix[total_goals >= 3].sum()),
        "Under 2.5": float(matrix[total_goals <= 2].sum()),
    }


def no_vig_probabilities(prices: Dict[str, float]) -> Dict[str, float]:
    inv = {k: 1.0 / v for k, v in prices.items() if v and v > 1.01}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()} if total > 0 else {}


def market_consensus_probs(bookmakers: Sequence[dict], home_api: str, away_api: str) -> Dict[Tuple[str, str], float]:
    """Average bookmaker-level no-vig probabilities per market/tip."""
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            market_key = str(market.get("key", ""))
            prices: Dict[str, float] = {}
            market_name = ""
            for outcome in market.get("outcomes", []):
                label = outcome_label(market_key, str(outcome.get("name", "")), home_api, away_api, outcome.get("point"))
                if not label:
                    continue
                tip, market_name = label
                price = safe_float(outcome.get("price", 0))
                if price > 1.01:
                    prices[tip] = price
            # Only normalize complete markets; partial books distort vig removal.
            needed = {"1", "X", "2"} if market_key == "h2h" else {"Over 2.5", "Under 2.5"}
            if needed.issubset(set(prices)):
                for tip, prob in no_vig_probabilities(prices).items():
                    buckets.setdefault((market_name, tip), []).append(prob)
    return {k: float(np.mean(v)) for k, v in buckets.items() if len(v) >= MIN_CONSENSUS_BOOKMAKERS}


def shrink_probability(prob: float, shrink: Optional[float] = None) -> float:
    """Pull extreme probabilities toward 50% to reduce model overconfidence.

    Betting models often overstate edges. Shrinking the blended probability
    reduces oversized Kelly stakes and makes reported fair odds more realistic.
    """
    factor = min(max(PROB_SHRINK if shrink is None else shrink, 0.0), 1.0)
    return float(min(max(0.5 + (prob - 0.5) * factor, 0.01), 0.99))


def blended_probability(model_prob: float, market_prob: Optional[float]) -> float:
    # Default is intentionally market-heavy. The market is usually a better
    # prior than a lightweight goals model, especially in top European leagues.
    w = min(max(MARKET_BLEND_WEIGHT, 0.0), 0.85)
    if market_prob is None or market_prob <= 0:
        raw = model_prob
    else:
        raw = (1.0 - w) * model_prob + w * market_prob
    return shrink_probability(raw)


def confidence_penalty(prob_model: float, prob_market: Optional[float], odds: float, edge: float, bookmaker: str) -> float:
    """Penalty for fragile value signals. Higher means less confidence."""
    penalty = 0.0
    if prob_market is not None:
        penalty += max(0.0, abs(prob_model - prob_market) - 0.12) * 35
    penalty += max(0.0, edge - 0.10) * 28
    penalty += max(0.0, odds - 3.0) * 1.8
    soft_books = ("1xbet", "codere", "betano", "nordic", "unibet")
    if any(x in bookmaker.lower() for x in soft_books):
        penalty += 0.75
    return round(penalty, 4)


def build_team_strengths(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    avg_h = max(df["FTHG"].mean(), 0.05)
    avg_a = max(df["FTAG"].mean(), 0.05)
    teams = sorted(set(df["HomeTeam"].astype(str)) | set(df["AwayTeam"].astype(str)))
    strengths: Dict[str, Dict[str, float]] = {}

    for team in teams:
        home = df[df["HomeTeam"] == team].tail(RECENT_MATCHES)
        away = df[df["AwayTeam"] == team].tail(RECENT_MATCHES)
        all_home = df[df["HomeTeam"] == team]
        all_away = df[df["AwayTeam"] == team]

        def blended(series_recent: pd.Series, series_all: pd.Series, base: float) -> float:
            r = series_recent.mean() if len(series_recent) else base
            a = series_all.mean() if len(series_all) else base
            return float((0.65 * r + 0.35 * a) / base) if base else 1.0

        strengths[team] = {
            "attack_home": blended(home["FTHG"], all_home["FTHG"], avg_h),
            "defence_home": blended(home["FTAG"], all_home["FTAG"], avg_a),
            "attack_away": blended(away["FTAG"], all_away["FTAG"], avg_a),
            "defence_away": blended(away["FTHG"], all_away["FTHG"], avg_h),
            "avg_h": avg_h,
            "avg_a": avg_a,
        }
    return strengths


def expected_goals(home: str, away: str, strengths: Dict[str, Dict[str, float]], home_adv: float) -> Tuple[float, float]:
    h = strengths[home]
    a = strengths[away]
    lh = h["attack_home"] * a["defence_away"] * h["avg_h"] + home_adv
    la = a["attack_away"] * h["defence_home"] * h["avg_a"]
    return round(max(lh, 0.05), 4), round(max(la, 0.05), 4)


def train_ai_model() -> Tuple[Optional[Any], List[str]]:
    features = ["edge", "market_edge", "kurz", "lh", "la", "prob_model", "prob_final", "score"]
    if LGBMClassifier is None or CalibratedClassifierCV is None or StratifiedKFold is None:
        log.info("LightGBM/sklearn nie je dostupny - AI filter vypnuty.")
        return None, features
    try:
        with db_connect() as conn:
            df = pd.read_sql("SELECT * FROM bets WHERE result IN ('V','P')", conn)
        if len(df) < 120 or df["result"].nunique() < 2:
            log.info("Malo historie pre AI filter: %s tipov.", len(df))
            return None, features
        df["win"] = (df["result"] == "V").astype(int)
        for col in features:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        base = LGBMClassifier(
            n_estimators=160,
            learning_rate=0.035,
            max_depth=3,
            num_leaves=12,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=-1,
        )
        model = CalibratedClassifierCV(base, method="isotonic", cv=cv)
        model.fit(df[features], df["win"])
        log.info("AI filter natrenovany na %s vyhodnotenych tipoch.", len(df))
        return model, features
    except Exception as e:
        log.warning("AI trening zlyhal: %s", e)
        return None, features


# =============================== BET SCANNER ===============================

def outcome_label(market_key: str, outcome_name: str, home: str, away: str, point: Optional[float] = None) -> Optional[Tuple[str, str]]:
    if market_key == "h2h":
        if outcome_name == home:
            return "1", "h2h"
        if outcome_name == away:
            return "2", "h2h"
        if outcome_name.lower() == "draw":
            return "X", "h2h"
    if market_key == "totals":
        # The Odds API can return multiple total-goal lines. This model only
        # estimates Over/Under 2.5, so skip all other lines instead of silently
        # treating them as 2.5.
        if point is not None and abs(float(point) - 2.5) > 1e-9:
            return None
        lower = outcome_name.lower()
        if lower == "over" or "over 2.5" in lower:
            return "Over 2.5", "totals"
        if lower == "under" or "under 2.5" in lower:
            return "Under 2.5", "totals"
    return None


async def process_league(session: aiohttp.ClientSession, league: str, cfg: Dict[str, Any], model: Optional[Any], features: List[str]) -> List[BetCandidate]:
    hist_url = f"https://www.football-data.co.uk/mmz4281/{football_data_season_code()}/{cfg['csv']}.csv"
    df_hist = await fetch_csv(session, hist_url)
    if df_hist is None or len(df_hist) < 30:
        log.info("%s: malo historickych dat.", league)
        return []

    strengths = build_team_strengths(df_hist)
    teams = set(strengths.keys())
    odds_data = await fetch_odds(session, cfg["api"])
    if not odds_data:
        return []
    out: List[BetCandidate] = []
    current = now_utc()
    horizon = current + timedelta(hours=LOOKAHEAD_HOURS)

    for match in odds_data:
        try:
            home_api = str(match.get("home_team", ""))
            away_api = str(match.get("away_team", ""))
            home = match_team(home_api, teams)
            away = match_team(away_api, teams)
            if not home or not away:
                continue
            start = parse_utc(match["commence_time"])
            if not (current <= start <= horizon):
                continue
            if SNAPSHOT_ODDS:
                save_match_odds_snapshot(league, match, home, away, start.isoformat())

            lh, la = expected_goals(home, away, strengths, float(cfg["ha"]))
            probs = poisson_probs(lh, la, DIXON_COLES_RHO)
            consensus = market_consensus_probs(match.get("bookmakers", []), home_api, away_api)

            for bookmaker in match.get("bookmakers", []):
                bk_title = str(bookmaker.get("title", ""))
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")
                    for outcome in market.get("outcomes", []):
                        label = outcome_label(market_key, str(outcome.get("name", "")), home_api, away_api, outcome.get("point"))
                        if not label:
                            continue
                        tip, market_name = label
                        price = safe_float(outcome.get("price", 0))
                        if tip not in probs:
                            continue
                        prob_model = probs[tip]
                        prob_market = consensus.get((market_name, tip))
                        prob = blended_probability(prob_model, prob_market)
                        edge = (prob * price) - 1.0
                        market_edge = ((prob_model - prob_market) if prob_market is not None else None)
                        # Sanity guards: reject weak disagreement, extreme model/market gaps,
                        # and candidates without enough bookmaker consensus. These reduce fake value.
                        if prob_market is None:
                            continue
                        if market_edge is not None and market_edge < MIN_MARKET_EDGE:
                            continue
                        if abs(prob_model - prob_market) > MAX_MODEL_MARKET_GAP:
                            continue
                        if not is_reasonable_candidate(prob, price, edge):
                            continue
                        penalty = confidence_penalty(prob_model, prob_market, price, edge, bk_title)
                        stake = max(0.0, kelly_stake(prob, price) * max(0.35, 1.0 - penalty / 10.0))
                        stake = round(min(stake, BANK * MAX_STAKE_PCT), 2)
                        if stake < MIN_STAKE:
                            continue
                        base_score = value_score(prob, price, edge, lh, la) - penalty
                        ai_prob = None
                        score = base_score
                        if model is not None:
                            row = pd.DataFrame([{
                                "edge": edge,
                                "kurz": price,
                                "lh": lh,
                                "la": la,
                                "prob_model": prob_model,
                                "prob_final": prob,
                                "market_edge": market_edge or 0.0,
                                "score": base_score,
                            }])[features]
                            ai_prob = float(model.predict_proba(row)[0][1])
                            if ai_prob < MIN_AI_PROB:
                                continue
                            score = round(base_score + ai_prob * 10, 4)
                        datum_iso = start.isoformat()
                        datum_display = start.astimezone(LOCAL_TZ).strftime("%d.%m.%Y %H:%M")
                        source_hash = make_hash(datum_iso, league, home, away, tip, price, bk_title)
                        out.append(BetCandidate(
                            datum_iso=datum_iso,
                            datum_display=datum_display,
                            league=league,
                            zapas=f"{home} vs {away}",
                            home_team=home,
                            away_team=away,
                            tip=tip,
                            market=market_name,
                            kurz=round(price, 3),
                            prob_model=round(prob_model, 5),
                            prob_market=round(prob_market, 5) if prob_market is not None else None,
                            prob_final=round(prob, 5),
                            edge=round(edge, 5),
                            market_edge=round(market_edge, 5) if market_edge is not None else None,
                            lh=round(lh, 3),
                            la=round(la, 3),
                            vklad=stake,
                            bookmaker=bk_title,
                            ai_prob=round(ai_prob, 5) if ai_prob is not None else None,
                            score=round(score, 4),
                            source_hash=source_hash,
                        ))
        except Exception as e:
            log.debug("Preskoceny zapas pre chybu: %s", e)
            continue

    out.sort(key=lambda b: (b.score, b.edge), reverse=True)
    return out


def dedupe_best_odds(bets: Sequence[BetCandidate]) -> List[BetCandidate]:
    """Keep only the best candidate for each match/tip/market.

    Multiple bookmakers often produce the same value bet. Saving only the
    strongest quote makes reports cleaner and avoids over-counting exposure.
    """
    best: Dict[Tuple[str, str, str, str], BetCandidate] = {}
    for b in bets:
        key = (b.datum_iso, b.zapas, b.tip, b.market)
        cur = best.get(key)
        if cur is None or (b.kurz, b.score, b.edge) > (cur.kurz, cur.score, cur.edge):
            best[key] = b
    return sorted(best.values(), key=lambda b: (b.score, b.edge, b.kurz), reverse=True)



def apply_market_agreement_filter(bets: Sequence[BetCandidate]) -> List[BetCandidate]:
    """Require at least two bookmakers to show value for the same match/tip.

    Controlled by REQUIRE_MARKET_AGREEMENT=1. The scanner evaluates every
    bookmaker quote first, then this filter removes isolated single-bookmaker
    signals that are more likely to be stale or bad data.
    """
    if not REQUIRE_MARKET_AGREEMENT:
        return list(bets)

    grouped: Dict[Tuple[str, str, str, str], set[str]] = {}
    for b in bets:
        key = (b.datum_iso, b.zapas, b.tip, b.market)
        grouped.setdefault(key, set()).add(b.bookmaker)

    return [
        b for b in bets
        if len(grouped.get((b.datum_iso, b.zapas, b.tip, b.market), set())) >= MIN_BOOKMAKERS_AGREE
    ]


def apply_portfolio_limits(bets: Sequence[BetCandidate]) -> List[BetCandidate]:
    """Apply bankroll-level exposure controls after model ranking."""
    daily_cap = BANK * MAX_DAILY_EXPOSURE_PCT
    match_cap = BANK * MAX_MATCH_EXPOSURE_PCT
    league_cap = BANK * MAX_LEAGUE_EXPOSURE_PCT
    accepted: List[BetCandidate] = []
    used_total = 0.0
    used_match: Dict[str, float] = {}
    used_league: Dict[str, float] = {}

    for b in sorted(bets, key=lambda x: (x.score, x.edge, x.prob_model), reverse=True):
        if len(accepted) >= MAX_BETS_PER_DAY:
            break
        match_key = f"{b.datum_iso}|{b.zapas}"
        if used_total + b.vklad > daily_cap:
            continue
        if used_match.get(match_key, 0.0) + b.vklad > match_cap:
            continue
        if used_league.get(b.league, 0.0) + b.vklad > league_cap:
            continue
        accepted.append(b)
        used_total += b.vklad
        used_match[match_key] = used_match.get(match_key, 0.0) + b.vklad
        used_league[b.league] = used_league.get(b.league, 0.0) + b.vklad
    return accepted


def portfolio_summary(bets: Sequence[BetCandidate]) -> str:
    if not bets:
        return "Portfolio: 0 tipov | Exposure 0.00 EUR"
    exposure = sum(b.vklad for b in bets)
    ev = sum(b.ev_eur for b in bets)
    avg_edge = sum(b.edge for b in bets) / len(bets)
    return (
        f"Portfolio: {len(bets)} tipov | Exposure {exposure:.2f} EUR "
        f"({exposure / BANK:.1%} banku) | Odhad EV {ev:.2f} EUR | Avg edge {avg_edge:.1%}"
    )

def performance_summary() -> str:
    with db_connect() as conn:
        df = pd.read_sql("SELECT kurz, vklad, result, created_at FROM bets WHERE result IN ('V','P')", conn)
    if df.empty:
        return "Zatial nie je dost vyhodnotenych tipov na performance summary."
    df["kurz"] = pd.to_numeric(df["kurz"], errors="coerce").fillna(0.0)
    df["vklad"] = pd.to_numeric(df["vklad"], errors="coerce").fillna(0.0)
    df["profit"] = np.where(df["result"] == "V", df["vklad"] * (df["kurz"] - 1.0), -df["vklad"])
    turnover = float(df["vklad"].sum())
    profit = float(df["profit"].sum())
    wins = int((df["result"] == "V").sum())
    total = int(len(df))
    hit_rate = wins / total if total else 0.0
    yield_pct = profit / turnover if turnover else 0.0
    equity = df["profit"].cumsum()
    drawdown = float((equity.cummax() - equity).max()) if not equity.empty else 0.0
    return (
        f"Performance: {total} settled | Win rate {hit_rate:.1%} | "
        f"P/L {profit:.2f} EUR | Yield {yield_pct:.1%} | Max DD {drawdown:.2f} EUR"
    )


def backtest_report(days: int = BACKTEST_DAYS) -> str:
    # Lightweight DB-based backtest of already generated and settled selections.
    # True odds-history backtesting requires archived bookmaker odds, which this
    # script does not yet store before match start.
    with db_connect() as conn:
        df = pd.read_sql("SELECT * FROM bets WHERE result IN ('V','P') ORDER BY datum_iso", conn)
    if df.empty:
        return "Backtest: bez vyhodnotenych historickych tipov v databaze."
    if days > 0 and "created_at" in df:
        cutoff = datetime.now() - timedelta(days=days)
        dt = pd.to_datetime(df["created_at"], errors="coerce")
        df = df[dt.isna() | (dt >= cutoff)]
    if df.empty:
        return f"Backtest: za poslednych {days} dni nie su vyhodnotene tipy."
    df["kurz"] = pd.to_numeric(df["kurz"], errors="coerce").fillna(0.0)
    df["vklad"] = pd.to_numeric(df["vklad"], errors="coerce").fillna(0.0)
    df["profit"] = np.where(df["result"] == "V", df["vklad"] * (df["kurz"] - 1.0), -df["vklad"])
    total = len(df)
    turnover = df["vklad"].sum()
    profit = df["profit"].sum()
    by_market = df.groupby("market")["profit"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    lines = [
        f"Backtest poslednych {days} dni z ulozenych tipov:",
        f"Tipy: {total} | Obrat: {turnover:.2f} EUR | Profit: {profit:.2f} EUR | Yield: {(profit/turnover if turnover else 0):.1%}",
        "Podla marketu:",
    ]
    for market, row in by_market.iterrows():
        lines.append(f"- {market or 'unknown'}: {int(row['count'])} tipov | {row['sum']:.2f} EUR")
    return "\n".join(lines)



# =============================== ANALYTICS / AUDIT ===============================

def _safe_col(df: pd.DataFrame, col: str, default: Any = np.nan) -> pd.Series:
    """Return a column even when the local SQLite DB was created by an older version."""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _format_group_table(df: pd.DataFrame, group_col: str, title: str, min_settled: int = 1, top_n: int = 12) -> List[str]:
    if df.empty or group_col not in df.columns:
        return [f"{title}: bez dat."]
    tmp = df.copy()
    tmp[group_col] = tmp[group_col].fillna("unknown").replace("", "unknown")
    grouped = []
    for key, g in tmp.groupby(group_col):
        settled = g[g["is_settled"]]
        turnover = float(settled["vklad"].sum())
        profit = float(settled["profit"].sum())
        clv = pd.to_numeric(g.get("clv_pct", pd.Series(dtype=float)), errors="coerce")
        grouped.append({
            group_col: key,
            "bets": int(len(g)),
            "settled": int(len(settled)),
            "profit": profit,
            "yield": profit / turnover if turnover else np.nan,
            "avg_clv": float(clv.mean()) if clv.notna().any() else np.nan,
            "clv_win": float((clv.dropna() > 0).mean()) if clv.notna().any() else np.nan,
            "avg_edge": float(pd.to_numeric(g.get("edge", pd.Series(dtype=float)), errors="coerce").mean()),
        })
    res = pd.DataFrame(grouped)
    if res.empty:
        return [f"{title}: bez dat."]
    res = res[(res["settled"] >= min_settled) | (res["bets"] >= min_settled)]
    res = res.sort_values(["profit", "avg_clv", "settled"], ascending=False).head(top_n)
    lines = [title]
    for _, r in res.iterrows():
        y = "n/a" if pd.isna(r["yield"]) else f"{r['yield']:.1%}"
        clv = "n/a" if pd.isna(r["avg_clv"]) else f"{r['avg_clv']:.2%}"
        clvw = "n/a" if pd.isna(r["clv_win"]) else f"{r['clv_win']:.1%}"
        edge = "n/a" if pd.isna(r["avg_edge"]) else f"{r['avg_edge']:.1%}"
        lines.append(
            f"- {r[group_col]}: bets {int(r['bets'])} | settled {int(r['settled'])} | "
            f"P/L {r['profit']:.2f} EUR | Yield {y} | Avg CLV {clv} | CLV+ {clvw} | Avg edge {edge}"
        )
    return lines


def calibration_report(df: pd.DataFrame) -> List[str]:
    settled = df[df["is_settled"]].copy()
    if settled.empty or "prob_final" not in settled.columns:
        return ["Calibration: bez vyhodnotenych tipov/prob_final."]
    settled["prob_final"] = pd.to_numeric(settled["prob_final"], errors="coerce")
    settled = settled.dropna(subset=["prob_final"])
    if settled.empty:
        return ["Calibration: bez platnych probabilit."]
    bins = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    settled["bucket"] = pd.cut(settled["prob_final"], bins=bins, include_lowest=True)
    lines = ["Calibration podľa P_final bucketov"]
    for bucket, g in settled.groupby("bucket", observed=True):
        if len(g) < 3:
            continue
        avg_p = float(g["prob_final"].mean())
        actual = float(g["win"].mean())
        diff = actual - avg_p
        lines.append(f"- {bucket}: n={len(g)} | avg P {avg_p:.1%} | real win {actual:.1%} | diff {diff:+.1%}")
    if len(lines) == 1:
        lines.append("- Málo dát v bucketoch; zbieraj väčšiu vzorku.")
    return lines


def analytics_report(days: int = ANALYTICS_DAYS) -> str:
    """Full audit report: performance, CLV, calibration and weak/strong segments."""
    init_db()
    with db_connect() as conn:
        df = pd.read_sql("SELECT * FROM bets ORDER BY created_at", conn)
        snapshots = pd.read_sql("SELECT * FROM odds_snapshots", conn)
    if df.empty:
        return "Analytics: databaza zatial neobsahuje ziadne tipy. Najprv spusti scanner alebo importuj historiu."

    if days > 0 and "created_at" in df.columns:
        cutoff = datetime.now() - timedelta(days=days)
        created = pd.to_datetime(df["created_at"], errors="coerce")
        df = df[created.isna() | (created >= cutoff)].copy()
    if df.empty:
        return f"Analytics: za poslednych {days} dni nie su v DB ziadne tipy."

    df["kurz"] = pd.to_numeric(_safe_col(df, "kurz", 0.0), errors="coerce").fillna(0.0)
    df["vklad"] = pd.to_numeric(_safe_col(df, "vklad", 0.0), errors="coerce").fillna(0.0)
    df["edge"] = pd.to_numeric(_safe_col(df, "edge", np.nan), errors="coerce")
    df["prob_final"] = pd.to_numeric(_safe_col(df, "prob_final", np.nan), errors="coerce")
    df["clv_pct"] = pd.to_numeric(_safe_col(df, "clv_pct", np.nan), errors="coerce")
    df["result"] = _safe_col(df, "result", "").fillna("").astype(str)
    df["is_settled"] = df["result"].isin(["V", "P"])
    df["win"] = (df["result"] == "V").astype(int)
    df["profit"] = np.where(df["result"] == "V", df["vklad"] * (df["kurz"] - 1.0), np.where(df["result"] == "P", -df["vklad"], 0.0))

    settled = df[df["is_settled"]].copy()
    open_bets = df[~df["is_settled"]].copy()
    turnover = float(settled["vklad"].sum())
    profit = float(settled["profit"].sum())
    total = int(len(settled))
    wins = int(settled["win"].sum()) if total else 0
    hit_rate = wins / total if total else 0.0
    yield_pct = profit / turnover if turnover else 0.0
    equity = settled["profit"].cumsum() if total else pd.Series(dtype=float)
    max_dd = float((equity.cummax() - equity).max()) if not equity.empty else 0.0
    profit_factor = float(settled.loc[settled["profit"] > 0, "profit"].sum() / abs(settled.loc[settled["profit"] < 0, "profit"].sum())) if total and abs(settled.loc[settled["profit"] < 0, "profit"].sum()) > 0 else np.nan
    brier = float(((settled["win"] - settled["prob_final"]) ** 2).mean()) if total and settled["prob_final"].notna().any() else np.nan
    avg_clv = float(df["clv_pct"].mean()) if df["clv_pct"].notna().any() else np.nan
    clv_win = float((df["clv_pct"].dropna() > 0).mean()) if df["clv_pct"].notna().any() else np.nan
    open_exposure = float(open_bets["vklad"].sum())

    lines = [
        f"ANALYTICS REPORT v{APP_VERSION} | obdobie: poslednych {days} dni",
        f"Tipy v DB: {len(df)} | Vyhodnotene: {total} | Otvorene: {len(open_bets)} | Open exposure: {open_exposure:.2f} EUR",
        f"P/L {profit:.2f} EUR | Turnover {turnover:.2f} EUR | Yield {yield_pct:.1%} | Win rate {hit_rate:.1%} | Max DD {max_dd:.2f} EUR",
        f"Profit factor: {'n/a' if pd.isna(profit_factor) else f'{profit_factor:.2f}'} | Brier: {'n/a' if pd.isna(brier) else f'{brier:.4f}'}",
        f"CLV: {'n/a' if pd.isna(avg_clv) else f'{avg_clv:.2%}'} avg | CLV+ rate: {'n/a' if pd.isna(clv_win) else f'{clv_win:.1%}'} | Odds snapshots: {len(snapshots)}",
        "",
    ]
    lines.extend(calibration_report(df))
    lines.append("")
    lines.extend(_format_group_table(df, "league", "Výkon podľa ligy"))
    lines.append("")
    lines.extend(_format_group_table(df, "market", "Výkon podľa marketu"))
    lines.append("")
    lines.extend(_format_group_table(df, "bookmaker", "Výkon podľa bookmakera"))
    lines.append("")
    lines.append("Interpretácia:")
    lines.append("- Priorita je kladné CLV na veľkej vzorke; krátkodobý profit môže byť len variance.")
    lines.append("- Segmenty s negatívnym CLV a slabým yieldom sú kandidáti na vypnutie alebo vyšší MIN_EDGE.")
    lines.append("- Ak calibration bucket dlhodobo nadhodnocuje real win rate, zníž MARKET_BLEND_WEIGHT alebo sprísni MIN_MARKET_EDGE.")
    return "\n".join(lines)


# =============================== REPORTING ===============================

def export_csv(bets: Sequence[BetCandidate]) -> Optional[Path]:
    if not bets:
        return None
    path = EXPORT_DIR / f"value_bets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields = [
        "datum_display", "league", "zapas", "tip", "market", "kurz", "prob_model",
        "prob_market", "prob_final", "edge", "market_edge", "implied_prob", "fair_odds", "ev_eur", "risk_level",
        "lh", "la", "vklad", "bookmaker", "ai_prob", "score"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for b in bets:
            writer.writerow({k: getattr(b, k) for k in fields})
    return path


def export_json(bets: Sequence[BetCandidate]) -> Optional[Path]:
    if not bets:
        return None
    path = EXPORT_DIR / f"value_bets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    rows = []
    for b in bets:
        d = {k: getattr(b, k) for k in b.__dataclass_fields__}
        d.update({
            "implied_prob": b.implied_prob,
            "fair_odds": b.fair_odds,
            "ev_eur": b.ev_eur,
            "risk_level": b.risk_level,
        })
        rows.append(d)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def format_report(bets: Sequence[BetCandidate], settled: int, inserted: int) -> str:
    lines = [
        f"PRODUCTION FOOTBALL BETTING MODEL v{APP_VERSION}",
        f"Bank: {BANK:.2f} EUR | Kelly fraction: {KELLY_FRAC} | Max stake: {MAX_STAKE_PCT*100:.1f}% | Market weight: {MARKET_BLEND_WEIGHT:.0%} | Shrink: {PROB_SHRINK:.0%}",
        f"Vyhodnotene stare tipy: {settled}",
        f"Nove ulozene tipy: {inserted}",
        performance_summary(),
        portfolio_summary(bets),
        "",
    ]
    if not bets:
        lines.append("Dnes bez value tipov podla aktualnych filtrov.")
        return "\n".join(lines)
    for i, b in enumerate(bets, 1):
        ai = f" | AI {b.ai_prob:.2%}" if b.ai_prob is not None else ""
        lines.append(
            f"{i:02d}. {b.datum_display} | {b.league} | {b.zapas} | {b.tip} @ {b.kurz} "
            f"| Fair {b.fair_odds} | P_final {b.prob_final:.1%} | P_model {b.prob_model:.1%} "
            f"| P_market {(b.prob_market or 0):.1%} | Edge {b.edge*100:.1f}% "
            f"| MEdge {((b.market_edge or 0)*100):.1f}% | EV {b.ev_eur:.2f} EUR "
            f"| Stake {b.vklad:.2f} EUR | Risk {b.risk_level} "
            f"| {b.bookmaker}{ai}"
        )
    return "\n".join(lines)


def send_email_report(subject: str, body: str) -> bool:
    if not (GMAIL_USER and GMAIL_PASSWORD and GMAIL_RECEIVER):
        return False
    msg = MIMEMultipart()
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.send_message(msg)
        return True
    except Exception as e:
        log.warning("Email report zlyhal: %s", e)
        return False


# =============================== MAIN ===============================

async def run() -> None:
    init_db()
    if ANALYTICS_ONLY:
        print(analytics_report(ANALYTICS_DAYS))
        return
    if BACKTEST_ONLY:
        print(backtest_report())
        return

    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        settled = await settle_results(session)
        if SETTLE_ONLY:
            print(f"Vyhodnotene tipy: {settled}\n{performance_summary()}")
            return
        model, features = train_ai_model()
        leagues = {k: v for k, v in LIGY.items() if ACTIVE_LEAGUES is None or k in ACTIVE_LEAGUES}
        tasks = [process_league(session, name, cfg, model, features) for name, cfg in leagues.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    bets: List[BetCandidate] = []
    for res in results:
        if isinstance(res, Exception):
            log.warning("Liga zlyhala: %s", res)
            continue
        bets.extend(res)
    bets = apply_market_agreement_filter(bets)
    bets = dedupe_best_odds(bets)
    bets = apply_portfolio_limits(bets)
    bets = bets[:TOP_N_REPORT]

    inserted = 0 if DRY_RUN else insert_bets(bets)
    clv_updated = 0 if DRY_RUN else update_clv_metrics()
    if clv_updated:
        log.info("Aktualizovane CLV metriky: %s", clv_updated)
    csv_path = export_csv(bets)
    json_path = export_json(bets)
    report = format_report(bets, settled, inserted)
    if DRY_RUN:
        report += "\n\nDRY RUN: tipy neboli ulozene do DB."
    print(report)
    if csv_path:
        log.info("CSV export: %s", csv_path)
    if json_path:
        log.info("JSON export: %s", json_path)
    if bets and not DRY_RUN and not NO_EMAIL:
        sent = send_email_report(f"Value Bets v{APP_VERSION} - {datetime.now(LOCAL_TZ).strftime('%d.%m %H:%M')}", report)
        if sent:
            log.info("Email report odoslany na %s", GMAIL_RECEIVER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Football value betting model {APP_VERSION}")
    parser.add_argument("--dry-run", action="store_true", help="Vypis tipy bez ulozenia do DB a bez emailu")
    parser.add_argument("--settle-only", action="store_true", help="Len vyhodnot stare tipy")
    parser.add_argument("--backtest", action="store_true", help="Zobraz DB backtest ulozenych vyhodnotenych tipov")
    parser.add_argument("--analytics", action="store_true", help="Zobraz audit modelu: CLV, calibration, ligy, markety, bookmakeri")
    parser.add_argument("--analytics-days", type=int, default=ANALYTICS_DAYS, help="Pocet dni pre analytics report")
    parser.add_argument("--backtest-days", type=int, default=BACKTEST_DAYS, help="Pocet dni pre DB backtest")
    parser.add_argument("--league", action="append", choices=sorted(LIGY.keys()), help="Obmedz sken na ligu; mozes zadat viackrat")
    parser.add_argument("--min-edge", type=float, help="Docasne prepis MIN_EDGE, napr. 0.06")
    parser.add_argument("--no-email", action="store_true", help="Neposielaj email report")
    parser.add_argument("--top", type=int, help="Max pocet tipov v reporte")
    parser.add_argument("--bank", type=float, help="Docasne prepis bank")
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global ACTIVE_LEAGUES, DRY_RUN, SETTLE_ONLY, NO_EMAIL, BACKTEST_ONLY, ANALYTICS_ONLY, ANALYTICS_DAYS, BACKTEST_DAYS, MIN_EDGE, TOP_N_REPORT, BANK
    ACTIVE_LEAGUES = set(args.league) if args.league else None
    DRY_RUN = bool(args.dry_run)
    SETTLE_ONLY = bool(args.settle_only)
    NO_EMAIL = bool(args.no_email or args.dry_run)
    BACKTEST_ONLY = bool(args.backtest)
    ANALYTICS_ONLY = bool(args.analytics)
    ANALYTICS_DAYS = int(args.analytics_days)
    BACKTEST_DAYS = int(args.backtest_days)
    if args.min_edge is not None:
        MIN_EDGE = float(args.min_edge)
    if args.top is not None:
        TOP_N_REPORT = int(args.top)
    if args.bank is not None:
        BANK = float(args.bank)


def main() -> None:
    try:
        apply_args(parse_args())
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Ukoncene pouzivatelom.")
    except Exception as e:
        log.exception("Fatalna chyba: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
