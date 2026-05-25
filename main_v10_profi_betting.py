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

VERZIA 10.7 PROFI SAFE:
- defaultne vypina AI filter, lebo posledny holdout yield bol negativny
- znizuje Kelly a max stake pre mensi drawdown
- sprisnuje edge, max odds, daily volume a longshot guard
- ciel: menej tipov, nizsia variance, lepsia sanca prezit zle obdobia
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
import re
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

APP_VERSION = "11.1-quant-pro-live-news-ecl"
DB_SCHEMA_VERSION = 4

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
KELLY_FRAC = env_float("KELLY_FRAC", 0.0075)  # PROFI SAFE: lower Kelly to reduce drawdown
MAX_STAKE_PCT = env_float("MAX_STAKE_PCT", 0.005)  # PROFI SAFE: max 0.5% bank per bet
MIN_STAKE = env_float("MIN_STAKE", 1.0)
MIN_EDGE = env_float("MIN_EDGE", 0.08)  # PROFI SAFE: stricter minimum edge
MAX_EDGE = env_float("MAX_EDGE", 0.13)  # PROFI SAFE: reject suspiciously huge edges

# AI filter settings. The AI layer is intentionally conservative: it is used
# only when there is enough settled history and when a holdout check does not
# show obvious overfitting.
AI_FILTER_ENABLED = env_int("AI_FILTER_ENABLED", 0)  # PROFI SAFE: disabled until holdout yield is stable positive
MIN_AI_PROB = env_float("MIN_AI_PROB", 0.54)
MIN_AI_EDGE = env_float("MIN_AI_EDGE", 0.015)
MIN_AI_TRAIN_ROWS = env_int("MIN_AI_TRAIN_ROWS", 180)
MIN_AI_TEST_ROWS = env_int("MIN_AI_TEST_ROWS", 40)
AI_TEST_FRACTION = env_float("AI_TEST_FRACTION", 0.25)
MAX_AI_BRIER = env_float("MAX_AI_BRIER", 0.285)
MIN_AI_HOLDOUT_YIELD = env_float("MIN_AI_HOLDOUT_YIELD", -0.03)
AI_SCORE_WEIGHT = env_float("AI_SCORE_WEIGHT", 12.0)
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

    # ===== FIFA =====
    "FIFA World Cup": {
        "csv": "WC",
        "api": "soccer_fifa_world_cup",
        "ha": 0.18
    },

    "World Cup Qualifiers Europe": {
        "csv": "WCQEU",
        "api": "soccer_fifa_world_cup_qualifiers_europe",
        "ha": 0.20
    },

    "World Cup Qualifiers South America": {
        "csv": "WCQSA",
        "api": "soccer_fifa_world_cup_qualifiers_south_america",
        "ha": 0.28
    },

    "Club World Cup": {
        "csv": "CWC",
        "api": "soccer_fifa_club_world_cup",
        "ha": 0.15
    },
}

ACTIVE_LEAGUES: Optional[set[str]] = None
DRY_RUN = False
SETTLE_ONLY = False
NO_EMAIL = False
BACKTEST_ONLY = False
ANALYTICS_ONLY = False
HEALTH_ONLY = False
COLLECT_ODDS_ONLY = False
COLLECTOR_LOOP = False
BANKROLL_SIM_ONLY = False
COLLECTOR_INTERVAL_SECONDS = env_int("COLLECTOR_INTERVAL_SECONDS", 300)
BANKROLL_SIM_PATHS = env_int("BANKROLL_SIM_PATHS", 5000)
BANKROLL_SIM_BETS = env_int("BANKROLL_SIM_BETS", 1000)
BANKROLL_SIM_STAKE_PCT = env_float("BANKROLL_SIM_STAKE_PCT", 0.005)
BANKROLL_RUIN_PCT = env_float("BANKROLL_RUIN_PCT", 0.50)
ANALYTICS_DAYS = env_int("ANALYTICS_DAYS", 365)
BACKTEST_DAYS = env_int("BACKTEST_DAYS", 120)
MIN_PROB = env_float("MIN_PROB", 0.18)
MAX_ODDS = env_float("MAX_ODDS", 4.50)  # PROFI SAFE: avoid high-variance longshots by default
TOP_N_REPORT = env_int("TOP_N_REPORT", 2)  # PROFI SAFE: only strongest picks in report
MAX_BETS_PER_DAY = env_int("MAX_BETS_PER_DAY", 2)  # PROFI SAFE: low daily volume
MAX_DAILY_EXPOSURE_PCT = env_float("MAX_DAILY_EXPOSURE_PCT", 0.05)
MAX_MATCH_EXPOSURE_PCT = env_float("MAX_MATCH_EXPOSURE_PCT", 0.015)
MAX_LEAGUE_EXPOSURE_PCT = env_float("MAX_LEAGUE_EXPOSURE_PCT", 0.035)
REQUIRE_MARKET_AGREEMENT = env_int("REQUIRE_MARKET_AGREEMENT", 1)  # safer default: at least 2 bookmakers for same pick
MIN_BOOKMAKERS_AGREE = env_int("MIN_BOOKMAKERS_AGREE", 2)
DIXON_COLES_RHO = env_float("DIXON_COLES_RHO", -0.08)
MARKET_BLEND_WEIGHT = env_float("MARKET_BLEND_WEIGHT", 0.55)
MIN_MARKET_EDGE = env_float("MIN_MARKET_EDGE", 0.035)  # edge vs no-vig consensus, model sanity guard
SNAPSHOT_ODDS = env_int("SNAPSHOT_ODDS", 1)  # store raw odds snapshots for CLV tracking
CLV_LOOKBACK_HOURS = env_int("CLV_LOOKBACK_HOURS", 12)
PROB_SHRINK = env_float("PROB_SHRINK", 0.72)
MAX_MODEL_MARKET_GAP = env_float("MAX_MODEL_MARKET_GAP", 0.24)
MIN_CONSENSUS_BOOKMAKERS = env_int("MIN_CONSENSUS_BOOKMAKERS", 2)

# Drawdown-aware safety layer. If recent settled bets are losing, the scanner
# either cuts stakes or pauses new exposure. Values are conservative defaults
# and can be overridden in .env.
RISK_COOLDOWN_DAYS = env_int("RISK_COOLDOWN_DAYS", 14)
RISK_REDUCE_AFTER_LOSS_PCT = env_float("RISK_REDUCE_AFTER_LOSS_PCT", 0.025)
RISK_STOP_AFTER_LOSS_PCT = env_float("RISK_STOP_AFTER_LOSS_PCT", 0.06)
RISK_REDUCED_STAKE_MULT = env_float("RISK_REDUCED_STAKE_MULT", 0.50)

# Segment/CLV quality controls. These use only already-settled historical bets
# from your own DB, so weak combinations can be downweighted or blocked without
# hardcoding bookmaker/league opinions.
SEGMENT_LOOKBACK_DAYS = env_int("SEGMENT_LOOKBACK_DAYS", 365)
SEGMENT_MIN_SETTLED = env_int("SEGMENT_MIN_SETTLED", 10)
SEGMENT_BLOCK_YIELD = env_float("SEGMENT_BLOCK_YIELD", -0.12)
SEGMENT_BLOCK_CLV = env_float("SEGMENT_BLOCK_CLV", -0.025)
SEGMENT_REDUCE_YIELD = env_float("SEGMENT_REDUCE_YIELD", -0.04)
SEGMENT_REDUCE_CLV = env_float("SEGMENT_REDUCE_CLV", -0.010)
SEGMENT_REDUCED_STAKE_MULT = env_float("SEGMENT_REDUCED_STAKE_MULT", 0.50)
CLV_STAKE_REDUCE_MULT = env_float("CLV_STAKE_REDUCE_MULT", 0.65)
DRAW_MIN_EDGE = env_float("DRAW_MIN_EDGE", 0.085)
DRAW_MAX_ODDS = env_float("DRAW_MAX_ODDS", 4.20)
LONGSHOT_MAX_ODDS = env_float("LONGSHOT_MAX_ODDS", 4.00)  # PROFI SAFE: stricter longshot guard
LONGSHOT_MIN_EDGE = env_float("LONGSHOT_MIN_EDGE", 0.14)  # PROFI SAFE: longshots need much stronger edge
LONGSHOT_MIN_PROB = env_float("LONGSHOT_MIN_PROB", 0.22)

# Walk-forward and segment governance. These keep validation closer to real live use:
# past data can explain future decisions, but future data must never tune past picks.
WALK_FORWARD_MIN_TRAIN_ROWS = env_int("WALK_FORWARD_MIN_TRAIN_ROWS", 80)
WALK_FORWARD_MIN_TEST_ROWS = env_int("WALK_FORWARD_MIN_TEST_ROWS", 8)
SEGMENT_COMBO_MIN_SETTLED = env_int("SEGMENT_COMBO_MIN_SETTLED", max(SEGMENT_MIN_SETTLED * 2, 24))
SEGMENT_DISABLE_REQUIRE_CLV_SAMPLES = env_int("SEGMENT_DISABLE_REQUIRE_CLV_SAMPLES", 8)

# Quant/meta layer. This does not replace the base model; it decides whether a
# value signal deserves full stake, reduced stake, or no bet. The strongest
# live signal is CLV/line movement; ELO is a second independent sanity check.
ELO_ENABLED = env_int("ELO_ENABLED", 1)
ELO_BASE = env_float("ELO_BASE", 1500.0)
ELO_K = env_float("ELO_K", 22.0)
ELO_HOME_ADV = env_float("ELO_HOME_ADV", 65.0)
META_LAYER_ENABLED = env_int("META_LAYER_ENABLED", 1)
META_MIN_CONFIDENCE = env_float("META_MIN_CONFIDENCE", 0.58)
META_REDUCE_BELOW = env_float("META_REDUCE_BELOW", 0.70)
META_REDUCED_STAKE_MULT = env_float("META_REDUCED_STAKE_MULT", 0.65)
LINE_MOVE_GOOD_PCT = env_float("LINE_MOVE_GOOD_PCT", 0.010)
LINE_MOVE_BAD_PCT = env_float("LINE_MOVE_BAD_PCT", -0.020)
CLV_MODEL_MIN_SAMPLES = env_int("CLV_MODEL_MIN_SAMPLES", 25)
CLV_MODEL_REDUCE_AVG = env_float("CLV_MODEL_REDUCE_AVG", -0.008)
CLV_MODEL_BLOCK_AVG = env_float("CLV_MODEL_BLOCK_AVG", -0.020)
FEATURE_STORE_ENABLED = env_int("FEATURE_STORE_ENABLED", 1)

# =============================== V11 QUANT-PRO MODULES ===============================
# These modules are designed to work with the existing odds_snapshots and bets tables.
# They do not guarantee profit; they make the scanner more market-aware and reduce
# fake value caused by stale/soft lines.
SHARP_WEIGHTING_ENABLED = env_int("SHARP_WEIGHTING_ENABLED", 1)
BOOKMAKER_GRADING_ENABLED = env_int("BOOKMAKER_GRADING_ENABLED", 1)
BAYESIAN_UPDATE_ENABLED = env_int("BAYESIAN_UPDATE_ENABLED", 1)
ENSEMBLE_MODEL_ENABLED = env_int("ENSEMBLE_MODEL_ENABLED", 1)
STEAM_DETECTION_ENABLED = env_int("STEAM_DETECTION_ENABLED", 1)
EXPECTED_CLV_ENABLED = env_int("EXPECTED_CLV_ENABLED", 1)
INJURY_NEWS_ENABLED = env_int("INJURY_NEWS_ENABLED", 0)  # optional hook; disabled unless you add a news/injury source
LIVE_BETTING_ENABLED = env_int("LIVE_BETTING_ENABLED", 0)  # safety default: pre-match only

SHARP_DEFAULT_WEIGHT = env_float("SHARP_DEFAULT_WEIGHT", 1.0)
SHARP_SOFT_WEIGHT = env_float("SHARP_SOFT_WEIGHT", 0.70)
SHARP_PINNACLE_WEIGHT = env_float("SHARP_PINNACLE_WEIGHT", 2.30)
SHARP_EXCHANGE_WEIGHT = env_float("SHARP_EXCHANGE_WEIGHT", 1.80)
BOOKMAKER_GRADE_MIN_SAMPLES = env_int("BOOKMAKER_GRADE_MIN_SAMPLES", 20)
BOOKMAKER_GRADE_CLV_WEIGHT = env_float("BOOKMAKER_GRADE_CLV_WEIGHT", 8.0)
BOOKMAKER_GRADE_YIELD_WEIGHT = env_float("BOOKMAKER_GRADE_YIELD_WEIGHT", 2.5)

BAYES_MARKET_PRIOR_STRENGTH = env_float("BAYES_MARKET_PRIOR_STRENGTH", 28.0)
BAYES_MODEL_EVIDENCE_STRENGTH = env_float("BAYES_MODEL_EVIDENCE_STRENGTH", 10.0)
ENSEMBLE_MODEL_WEIGHT = env_float("ENSEMBLE_MODEL_WEIGHT", 0.34)
ENSEMBLE_MARKET_WEIGHT = env_float("ENSEMBLE_MARKET_WEIGHT", 0.36)
ENSEMBLE_BAYES_WEIGHT = env_float("ENSEMBLE_BAYES_WEIGHT", 0.20)
ENSEMBLE_ECL_WEIGHT = env_float("ENSEMBLE_ECL_WEIGHT", 0.10)

ODDS_VELOCITY_LOOKBACK_HOURS = env_int("ODDS_VELOCITY_LOOKBACK_HOURS", 24)
STEAM_MIN_BOOKMAKERS = env_int("STEAM_MIN_BOOKMAKERS", 3)
STEAM_MIN_MOVE_PCT = env_float("STEAM_MIN_MOVE_PCT", 0.012)
STEAM_BLOCK_BAD_MOVE_PCT = env_float("STEAM_BLOCK_BAD_MOVE_PCT", -0.025)
STEAM_SCORE_WEIGHT = env_float("STEAM_SCORE_WEIGHT", 0.10)
EXPECTED_CLV_MIN = env_float("EXPECTED_CLV_MIN", -0.006)
EXPECTED_CLV_STRONG = env_float("EXPECTED_CLV_STRONG", 0.010)
EXPECTED_CLV_SCORE_WEIGHT = env_float("EXPECTED_CLV_SCORE_WEIGHT", 45.0)

INJURY_NEWS_RISK_BLOCK = env_float("INJURY_NEWS_RISK_BLOCK", 0.75)
INJURY_NEWS_RISK_REDUCE = env_float("INJURY_NEWS_RISK_REDUCE", 0.45)
INJURY_STAKE_REDUCE_MULT = env_float("INJURY_STAKE_REDUCE_MULT", 0.60)
INJURY_NEWS_SOURCE = os.getenv("INJURY_NEWS_SOURCE", "").strip()  # local .json/.csv or URL returning JSON
INJURY_NEWS_CACHE_TTL_SECONDS = env_int("INJURY_NEWS_CACHE_TTL_SECONDS", 900)
INJURY_NEWS_MAX_AGE_HOURS = env_int("INJURY_NEWS_MAX_AGE_HOURS", 72)
INJURY_NEGATIVE_KEYWORDS = [x.strip().lower() for x in os.getenv(
    "INJURY_NEGATIVE_KEYWORDS",
    "injured,injury,out,doubtful,questionable,suspended,ban,illness,hamstring,knee,ankle,goalkeeper out,rested"
).split(",") if x.strip()]
INJURY_POSITIVE_KEYWORDS = [x.strip().lower() for x in os.getenv(
    "INJURY_POSITIVE_KEYWORDS",
    "returns,available,fit,back in training,cleared,expected to start"
).split(",") if x.strip()]

LIVE_MAX_MATCH_MINUTES = env_int("LIVE_MAX_MATCH_MINUTES", 75)
LIVE_MIN_EDGE = env_float("LIVE_MIN_EDGE", 0.10)
LIVE_MAX_ODDS = env_float("LIVE_MAX_ODDS", 3.25)
LIVE_STAKE_MULT = env_float("LIVE_STAKE_MULT", 0.35)
LIVE_REQUIRE_GOOD_STEAM = env_int("LIVE_REQUIRE_GOOD_STEAM", 1)
LIVE_MIN_STEAM_SCORE = env_float("LIVE_MIN_STEAM_SCORE", 0.20)
LIVE_BLOCK_WITHOUT_VELOCITY = env_int("LIVE_BLOCK_WITHOUT_VELOCITY", 1)

# Runtime cache populated once per run. The format is intentionally simple so it
# can be fed from your own scraper/API without changing the betting logic.
INJURY_NEWS_CACHE: List[Dict[str, Any]] = []
INJURY_NEWS_CACHE_AT: Optional[datetime] = None



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
    elo_diff: Optional[float] = None
    line_move_pct: Optional[float] = None
    meta_confidence: Optional[float] = None
    sharp_weight: Optional[float] = None
    bookmaker_grade: Optional[float] = None
    odds_velocity_pct_per_hour: Optional[float] = None
    steam_score: Optional[float] = None
    predicted_closing_odds: Optional[float] = None
    expected_clv_pct: Optional[float] = None
    bayes_prob: Optional[float] = None
    ensemble_prob: Optional[float] = None
    injury_news_risk: Optional[float] = None

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
        ensure_syndicate_schema(conn)
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_store (
            source_hash TEXT PRIMARY KEY,
            decision_at TEXT NOT NULL,
            datum_iso TEXT,
            league TEXT,
            home_team TEXT,
            away_team TEXT,
            tip TEXT,
            market TEXT,
            bookmaker TEXT,
            kurz REAL,
            prob_model REAL,
            prob_market REAL,
            prob_final REAL,
            edge REAL,
            market_edge REAL,
            lh REAL,
            la REAL,
            elo_diff REAL,
            line_move_pct REAL,
            meta_confidence REAL,
            ai_prob REAL,
            score REAL,
            app_version TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feature_store_decision ON feature_store(decision_at, league, market, tip)")


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
            kickoff = pd.to_datetime(datum_iso, errors="coerce", utc=True, format="ISO8601")
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




# =============================== V12 SYNDICATE INFRA ===============================

def ensure_syndicate_schema(conn: sqlite3.Connection) -> None:
    """Extra operational tables for pro workflow: collector state, decisions and simulations."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS collector_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            mode TEXT,
            leagues INTEGER DEFAULT 0,
            snapshots_saved INTEGER DEFAULT 0,
            errors INTEGER DEFAULT 0,
            note TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            source_hash TEXT,
            decision TEXT,
            reason TEXT,
            league TEXT,
            home_team TEXT,
            away_team TEXT,
            tip TEXT,
            market TEXT,
            bookmaker TEXT,
            price REAL,
            prob_final REAL,
            edge REAL,
            expected_clv_pct REAL,
            steam_score REAL,
            bookmaker_grade REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_audit_created ON decision_audit(created_at, decision)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bankroll_simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            paths INTEGER,
            bets INTEGER,
            start_bank REAL,
            stake_pct REAL,
            ruin_pct REAL,
            median_final REAL,
            p05_final REAL,
            p95_final REAL,
            prob_ruin REAL,
            median_max_dd REAL,
            note TEXT
        )
    """)


def runtime_health_report() -> str:
    """Operational checklist before real money / scheduled runs."""
    init_db()
    with db_connect() as conn:
        bets = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
        settled = conn.execute("SELECT COUNT(*) FROM bets WHERE result IN ('V','P')").fetchone()[0]
        open_bets = conn.execute("SELECT COUNT(*) FROM bets WHERE result IS NULL OR result='' ").fetchone()[0]
        snapshots = conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
        recent_snapshots = conn.execute(
            "SELECT COUNT(*) FROM odds_snapshots WHERE captured_at >= ?",
            ((now_utc() - timedelta(hours=24)).isoformat(),),
        ).fetchone()[0]
        runs = conn.execute("SELECT COUNT(*) FROM collector_runs").fetchone()[0]
    checks = []
    checks.append(("ODDS_API_KEY", bool(API_ODDS_KEY), "required for odds collection"))
    checks.append(("snapshots_24h", recent_snapshots > 0, f"{recent_snapshots} snapshots in last 24h"))
    checks.append(("settled_sample", settled >= 500, f"{settled} settled bets; 500+ preferred"))
    checks.append(("ai_filter", AI_FILTER_ENABLED == 0 or settled >= MIN_AI_TRAIN_ROWS, f"AI enabled={AI_FILTER_ENABLED}, min rows={MIN_AI_TRAIN_ROWS}"))
    checks.append(("injury_news", (not INJURY_NEWS_ENABLED) or bool(INJURY_NEWS_SOURCE), f"enabled={INJURY_NEWS_ENABLED}, source={INJURY_NEWS_SOURCE or 'none'}"))
    checks.append(("live_mode", not LIVE_BETTING_ENABLED, "live should stay disabled unless you have realtime feed/execution"))
    lines = [
        f"RUNTIME HEALTH REPORT v{APP_VERSION}",
        f"DB: {DB_FILE}",
        f"Bets: {bets} | Settled: {settled} | Open: {open_bets} | Odds snapshots: {snapshots} | Collector runs: {runs}",
        f"Bank: {BANK:.2f} EUR | Kelly {KELLY_FRAC} | Max stake {MAX_STAKE_PCT:.2%} | Top N {TOP_N_REPORT}",
        "",
        "Checks:",
    ]
    for name, ok, msg in checks:
        lines.append(f"- {'OK' if ok else 'WARN'} {name}: {msg}")
    lines.extend([
        "",
        "Recommended production cadence:",
        f"- Odds collector: python {Path(__file__).name} --collect-odds --collector-loop --collector-interval {COLLECTOR_INTERVAL_SECONDS}",
        f"- Scanner: python {Path(__file__).name} --dry-run --no-email",
        f"- Analytics: python {Path(__file__).name} --analytics",
        f"- Bankroll sim: python {Path(__file__).name} --bankroll-sim",
    ])
    return "\n".join(lines)


async def collect_odds_snapshots(session: aiohttp.ClientSession, loop_once: bool = True, interval_seconds: int = COLLECTOR_INTERVAL_SECONDS) -> str:
    """Collect bookmaker odds snapshots without creating tips.

    This is the missing 24/7 infra piece for steam, velocity and expected CLV.
    Run it from cron/systemd/GitHub Actions. The scanner gets stronger as this
    table grows because it can compare first/current/closing prices.
    """
    init_db()
    total_saved = 0
    total_errors = 0
    total_runs = 0
    while True:
        started = now_utc().isoformat()
        saved = 0
        errors = 0
        leagues_seen = 0
        for league, cfg in LIGY.items():
            if ACTIVE_LEAGUES is not None and league not in ACTIVE_LEAGUES:
                continue
            leagues_seen += 1
            try:
                odds_data = await fetch_odds(session, cfg["api"])
                saved += save_odds_snapshots(league, odds_data)
            except Exception as exc:
                errors += 1
                log.warning("Collector error %s: %s", league, exc)
        finished = now_utc().isoformat()
        total_saved += saved
        total_errors += errors
        total_runs += 1
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO collector_runs(started_at, finished_at, mode, leagues, snapshots_saved, errors, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (started, finished, "loop" if not loop_once else "once", leagues_seen, saved, errors, "odds snapshot collection"),
            )
        log.info("Collector run saved=%s errors=%s leagues=%s", saved, errors, leagues_seen)
        if loop_once:
            break
        await asyncio.sleep(max(30, int(interval_seconds)))
    return f"Odds collector finished: runs={total_runs}, saved_snapshots={total_saved}, errors={total_errors}"


def bankroll_monte_carlo_report(paths: int = BANKROLL_SIM_PATHS, bets: int = BANKROLL_SIM_BETS, stake_pct: float = BANKROLL_SIM_STAKE_PCT) -> str:
    """Monte Carlo bankroll simulator from your own settled bet return distribution."""
    init_db()
    with db_connect() as conn:
        df = pd.read_sql("SELECT kurz, result FROM bets WHERE result IN ('V','P') AND kurz > 1.01", conn)
    if len(df) < 50:
        return "Bankroll sim: minimum 50 settled bets required. Current settled sample is too small."
    df["kurz"] = pd.to_numeric(df["kurz"], errors="coerce").fillna(0.0)
    returns = np.where(df["result"] == "V", df["kurz"] - 1.0, -1.0)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 50:
        return "Bankroll sim: not enough clean returns after filtering."
    rng = np.random.default_rng(42)
    final_banks = []
    max_dds = []
    ruined = 0
    ruin_level = BANK * BANKROLL_RUIN_PCT
    for _ in range(int(paths)):
        bank = float(BANK)
        peak = bank
        max_dd = 0.0
        sample = rng.choice(returns, size=int(bets), replace=True)
        for r in sample:
            stake = max(MIN_STAKE, bank * float(stake_pct))
            stake = min(stake, bank * MAX_STAKE_PCT, bank)
            bank += stake * float(r)
            peak = max(peak, bank)
            max_dd = max(max_dd, peak - bank)
            if bank <= 0:
                bank = 0.0
                break
        if bank <= ruin_level:
            ruined += 1
        final_banks.append(bank)
        max_dds.append(max_dd)
    arr = np.array(final_banks)
    dd = np.array(max_dds)
    median_final = float(np.percentile(arr, 50))
    p05_final = float(np.percentile(arr, 5))
    p95_final = float(np.percentile(arr, 95))
    prob_ruin = ruined / max(1, int(paths))
    median_dd = float(np.percentile(dd, 50))
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO bankroll_simulations(paths, bets, start_bank, stake_pct, ruin_pct, median_final, p05_final, p95_final, prob_ruin, median_max_dd, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (int(paths), int(bets), BANK, float(stake_pct), BANKROLL_RUIN_PCT, median_final, p05_final, p95_final, prob_ruin, median_dd, "bootstrap from settled bets"),
        )
    hit_rate = float((returns > 0).mean())
    avg_ret = float(np.mean(returns))
    lines = [
        f"MONTE CARLO BANKROLL SIM v{APP_VERSION}",
        f"Input sample: {len(returns)} settled bets | Historical hit rate {hit_rate:.1%} | Avg unit return {avg_ret:.3f}",
        f"Paths: {int(paths)} | Bets/path: {int(bets)} | Start bank: {BANK:.2f} EUR | Stake pct: {float(stake_pct):.2%}",
        f"Final bank median: {median_final:.2f} EUR | p05: {p05_final:.2f} EUR | p95: {p95_final:.2f} EUR",
        f"Risk of falling below {BANKROLL_RUIN_PCT:.0%} bank: {prob_ruin:.1%} | Median max DD: {median_dd:.2f} EUR",
        "Interpretation: if p05 or ruin probability is uncomfortable, reduce BANKROLL_SIM_STAKE_PCT, KELLY_FRAC and MAX_STAKE_PCT before staking real money.",
    ]
    return "\n".join(lines)

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
                parsed_dt = pd.to_datetime(datum_iso, errors="coerce", utc=True, format="ISO8601")
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
            needed = {"1", "X", "2"} if market_key == "h2h" else {"Over 2.5", "Under 2.5"}
            if needed.issubset(set(prices)):
                for tip, prob in no_vig_probabilities(prices).items():
                    buckets.setdefault((market_name, tip), []).append(prob)
    return {k: float(np.mean(v)) for k, v in buckets.items() if len(v) >= MIN_CONSENSUS_BOOKMAKERS}


def bookmaker_sharp_weight(bookmaker: str) -> float:
    """Static prior for how much a bookmaker should influence consensus."""
    if not SHARP_WEIGHTING_ENABLED:
        return 1.0
    name = bookmaker.lower()
    if "pinnacle" in name or "pinny" in name:
        return SHARP_PINNACLE_WEIGHT
    if "betfair" in name or "matchbook" in name or "exchange" in name:
        return SHARP_EXCHANGE_WEIGHT
    soft_markers = ("unibet", "betano", "1xbet", "casumo", "nordic", "coolbet", "betsson", "leovegas", "william hill")
    if any(x in name for x in soft_markers):
        return SHARP_SOFT_WEIGHT
    return SHARP_DEFAULT_WEIGHT


def bookmaker_grade(bookmaker: str) -> float:
    """Dynamic 0.5-1.5 grade from own historical CLV/yield for this bookmaker."""
    if not BOOKMAKER_GRADING_ENABLED:
        return 1.0
    try:
        with db_connect() as conn:
            df = pd.read_sql(
                """
                SELECT kurz, vklad, result, clv_pct FROM bets
                WHERE bookmaker=? AND result IN ('V','P')
                """, conn, params=(bookmaker,)
            )
        if len(df) < BOOKMAKER_GRADE_MIN_SAMPLES:
            return 1.0
        df["kurz"] = pd.to_numeric(df["kurz"], errors="coerce").fillna(0.0)
        df["vklad"] = pd.to_numeric(df["vklad"], errors="coerce").fillna(0.0)
        profit = np.where(df["result"] == "V", df["vklad"] * (df["kurz"] - 1.0), -df["vklad"])
        turnover = float(df["vklad"].sum())
        yld = float(profit.sum() / turnover) if turnover > 0 else 0.0
        clv = float(pd.to_numeric(df["clv_pct"], errors="coerce").dropna().mean()) if df["clv_pct"].notna().any() else 0.0
        grade = 1.0 + (clv * BOOKMAKER_GRADE_CLV_WEIGHT) + (yld * BOOKMAKER_GRADE_YIELD_WEIGHT)
        return round(float(min(max(grade, 0.50), 1.50)), 4)
    except Exception:
        return 1.0


def market_consensus_probs_weighted(bookmakers: Sequence[dict], home_api: str, away_api: str) -> Dict[Tuple[str, str], float]:
    """Sharp-weighted no-vig market consensus. Pinnacle/exchanges count more, soft books less."""
    buckets: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for bookmaker in bookmakers:
        bk = str(bookmaker.get("title", ""))
        weight = bookmaker_sharp_weight(bk) * bookmaker_grade(bk)
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
            needed = {"1", "X", "2"} if market_key == "h2h" else {"Over 2.5", "Under 2.5"}
            if needed.issubset(set(prices)):
                for tip, prob in no_vig_probabilities(prices).items():
                    buckets.setdefault((market_name, tip), []).append((prob, weight))
    out: Dict[Tuple[str, str], float] = {}
    for key, vals in buckets.items():
        if len(vals) < MIN_CONSENSUS_BOOKMAKERS:
            continue
        probs = np.array([v[0] for v in vals], dtype=float)
        weights = np.array([v[1] for v in vals], dtype=float)
        out[key] = float(np.average(probs, weights=weights)) if weights.sum() > 0 else float(probs.mean())
    return out


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


AI_FEATURES = [
    "edge", "market_edge", "kurz", "implied_prob", "lh", "la", "total_lambda",
    "goal_diff", "prob_model", "prob_market", "prob_final", "model_market_gap",
    "odds_edge_ratio", "score", "elo_diff", "line_move_pct", "meta_confidence",
    "is_home_pick", "is_draw_pick", "is_away_pick", "is_total_pick",
]


def build_ai_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable numeric features for the AI filter.

    Keep this function shared between training and live prediction so the model
    sees exactly the same feature definitions in both paths.
    """
    out = df.copy()
    for col in ["edge", "market_edge", "kurz", "lh", "la", "prob_model", "prob_market", "prob_final", "score", "elo_diff", "line_move_pct", "meta_confidence"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["implied_prob"] = np.where(out["kurz"] > 1.01, 1.0 / out["kurz"], 0.0)
    out["total_lambda"] = out["lh"] + out["la"]
    out["goal_diff"] = out["lh"] - out["la"]
    out["model_market_gap"] = (out["prob_model"] - out["prob_market"]).abs()
    out["odds_edge_ratio"] = np.where(out["implied_prob"] > 0, out["edge"] / out["implied_prob"], 0.0)
    tip = out.get("tip", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
    market = out.get("market", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
    out["is_home_pick"] = (tip == "1").astype(int)
    out["is_draw_pick"] = (tip == "X").astype(int)
    out["is_away_pick"] = (tip == "2").astype(int)
    out["is_total_pick"] = (market == "totals").astype(int)
    return out


def ai_candidate_row(
    edge: float,
    market_edge: Optional[float],
    price: float,
    lh: float,
    la: float,
    prob_model: float,
    prob_market: Optional[float],
    prob_final: float,
    base_score: float,
    tip: str,
    market_name: str,
    elo_diff: Optional[float] = None,
    line_move_pct: Optional[float] = None,
    meta_confidence: Optional[float] = None,
) -> pd.DataFrame:
    row = pd.DataFrame([{
        "edge": edge,
        "market_edge": market_edge or 0.0,
        "kurz": price,
        "lh": lh,
        "la": la,
        "prob_model": prob_model,
        "prob_market": prob_market or 0.0,
        "prob_final": prob_final,
        "score": base_score,
        "elo_diff": elo_diff or 0.0,
        "line_move_pct": line_move_pct or 0.0,
        "meta_confidence": meta_confidence or 0.0,
        "tip": tip,
        "market": market_name,
    }])
    return build_ai_features(row)[AI_FEATURES]


def train_ai_model() -> Tuple[Optional[Any], List[str]]:
    features = AI_FEATURES
    if not AI_FILTER_ENABLED:
        log.info("AI filter vypnuty cez AI_FILTER_ENABLED=0.")
        return None, features
    if LGBMClassifier is None or CalibratedClassifierCV is None or StratifiedKFold is None:
        log.info("LightGBM/sklearn nie je dostupny - AI filter vypnuty.")
        return None, features
    try:
        with db_connect() as conn:
            df = pd.read_sql("SELECT * FROM bets WHERE result IN ('V','P') ORDER BY COALESCE(datum_iso, created_at)", conn)
        if len(df) < MIN_AI_TRAIN_ROWS or df["result"].nunique() < 2:
            log.info("Malo historie pre AI filter: %s tipov, minimum je %s.", len(df), MIN_AI_TRAIN_ROWS)
            return None, features

        df = build_ai_features(df)
        df["win"] = (df["result"] == "V").astype(int)
        df["profit"] = np.where(df["win"] == 1, df["vklad"] * (df["kurz"] - 1.0), -df["vklad"])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["win"])
        if len(df) < MIN_AI_TRAIN_ROWS or df["win"].nunique() < 2:
            log.info("AI filter nema dost cistych dat po feature engineeringu: %s.", len(df))
            return None, features

        test_size = max(MIN_AI_TEST_ROWS, int(len(df) * min(max(AI_TEST_FRACTION, 0.10), 0.40)))
        test_size = min(test_size, max(20, len(df) // 3))
        train_df = df.iloc[:-test_size].copy()
        test_df = df.iloc[-test_size:].copy()
        if len(train_df) < 80 or train_df["win"].nunique() < 2 or test_df["win"].nunique() < 2:
            log.info("AI holdout split nema dost tried v train/test - filter vypnuty.")
            return None, features

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        base = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=10,
            min_child_samples=18,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.15,
            reg_lambda=0.35,
            random_state=42,
            verbosity=-1,
        )
        method = "isotonic" if len(train_df) >= 350 else "sigmoid"
        model = CalibratedClassifierCV(base, method=method, cv=cv)
        model.fit(train_df[features], train_df["win"])

        holdout_prob = model.predict_proba(test_df[features])[:, 1]
        holdout_brier = float(np.mean((test_df["win"].to_numpy() - holdout_prob) ** 2))
        selected = test_df[(holdout_prob >= MIN_AI_PROB) & ((holdout_prob * test_df["kurz"] - 1.0) >= MIN_AI_EDGE)]
        selected_turnover = float(selected["vklad"].sum())
        selected_yield = float(selected["profit"].sum() / selected_turnover) if selected_turnover > 0 else 0.0

        if holdout_brier > MAX_AI_BRIER:
            log.info("AI filter vypnuty: holdout Brier %.4f > limit %.4f.", holdout_brier, MAX_AI_BRIER)
            return None, features
        if len(selected) >= 10 and selected_yield < MIN_AI_HOLDOUT_YIELD:
            log.info("AI filter vypnuty: holdout yield %.1f%% < limit %.1f%%.", selected_yield * 100, MIN_AI_HOLDOUT_YIELD * 100)
            return None, features

        # Refit on all settled data after the holdout sanity check passes.
        final_model = CalibratedClassifierCV(base, method=method, cv=cv)
        final_model.fit(df[features], df["win"])
        log.info(
            "AI filter aktivny: train=%s, holdout=%s, Brier=%.4f, selected=%s, selected_yield=%.1f%%, method=%s.",
            len(train_df), len(test_df), holdout_brier, len(selected), selected_yield * 100, method,
        )
        return final_model, features
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



# =============================== QUANT / META LAYER ===============================

def build_elo_ratings(df: pd.DataFrame) -> Dict[str, float]:
    """Build simple pre-match ELO ratings from historical final scores."""
    if not ELO_ENABLED or df is None or df.empty:
        return {}
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required.issubset(df.columns):
        return {}
    hist = df.dropna(subset=list(required)).copy()
    if "Date" in hist.columns:
        hist["_date"] = pd.to_datetime(hist["Date"], dayfirst=True, errors="coerce")
        hist = hist.sort_values("_date")
    ratings: Dict[str, float] = {}
    for _, r in hist.iterrows():
        home = str(r["HomeTeam"])
        away = str(r["AwayTeam"])
        hg = int(safe_float(r["FTHG"], 0))
        ag = int(safe_float(r["FTAG"], 0))
        rh = ratings.get(home, ELO_BASE)
        ra = ratings.get(away, ELO_BASE)
        expected_h = 1.0 / (1.0 + 10.0 ** (-((rh + ELO_HOME_ADV) - ra) / 400.0))
        actual_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margin = abs(hg - ag)
        margin_mult = 1.0 + min(margin, 4) * 0.15
        delta = ELO_K * margin_mult * (actual_h - expected_h)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings


def elo_diff_for_match(home: str, away: str, ratings: Dict[str, float]) -> float:
    if not ratings:
        return 0.0
    return round((ratings.get(home, ELO_BASE) + ELO_HOME_ADV) - ratings.get(away, ELO_BASE), 2)


def line_movement_pct(league: str, home: str, away: str, tip: str, market: str, bookmaker: str, current_price: float) -> Optional[float]:
    """Return positive value when current odds are better than previous stored odds.

    It is a proxy for whether the line is moving with or against us. Because odds
    snapshots are only available after the script has been running, absence of
    data returns None and never blocks a candidate by itself.
    """
    try:
        with db_connect() as conn:
            q = conn.execute(
                """
                SELECT price FROM odds_snapshots
                WHERE league=? AND home_team=? AND away_team=? AND tip=? AND market=? AND bookmaker=?
                ORDER BY captured_at ASC LIMIT 1
                """,
                (league, home, away, tip, market, bookmaker),
            ).fetchone()
        if not q:
            return None
        first_price = safe_float(q[0], 0.0)
        if first_price <= 1.01 or current_price <= 1.01:
            return None
        return round((current_price / first_price) - 1.0, 5)
    except Exception:
        return None


def historical_clv_quality(candidate: BetCandidate, min_samples: int = CLV_MODEL_MIN_SAMPLES) -> Tuple[float, str]:
    """Reduce or block if similar historical picks fail to beat closing line."""
    try:
        with db_connect() as conn:
            df = pd.read_sql(
                """
                SELECT clv_pct FROM bets
                WHERE result IN ('V','P')
                  AND league=? AND market=? AND tip=?
                  AND clv_pct IS NOT NULL
                """,
                conn,
                params=(candidate.league, candidate.market, candidate.tip),
            )
        if len(df) < min_samples:
            return 1.0, f"CLV model neutral: only {len(df)} similar samples"
        avg_clv = float(pd.to_numeric(df["clv_pct"], errors="coerce").dropna().mean())
        if avg_clv <= CLV_MODEL_BLOCK_AVG:
            return 0.0, f"CLV model block: avg CLV {avg_clv:.2%} on {len(df)} samples"
        if avg_clv <= CLV_MODEL_REDUCE_AVG:
            return CLV_STAKE_REDUCE_MULT, f"CLV model reduce: avg CLV {avg_clv:.2%} on {len(df)} samples"
        return 1.0, f"CLV model ok: avg CLV {avg_clv:.2%} on {len(df)} samples"
    except Exception as e:
        log.debug("CLV quality unavailable: %s", e)
        return 1.0, "CLV model unavailable"


def odds_velocity_metrics(league: str, home: str, away: str, tip: str, market: str, current_price: float) -> Dict[str, Optional[float]]:
    """Estimate odds movement speed, steam and predicted closing odds from stored snapshots.

    Positive velocity means the current offered price is higher than the earliest stored price.
    Negative velocity means the price has shortened. Steam score is positive when several
    bookmakers moved in the same direction strongly enough to matter.
    """
    metrics: Dict[str, Optional[float]] = {
        "velocity": None, "steam_score": None, "predicted_closing_odds": None, "expected_clv_pct": None
    }
    if not STEAM_DETECTION_ENABLED and not EXPECTED_CLV_ENABLED:
        return metrics
    try:
        cutoff = (now_utc() - timedelta(hours=ODDS_VELOCITY_LOOKBACK_HOURS)).isoformat()
        with db_connect() as conn:
            df = pd.read_sql(
                """
                SELECT bookmaker, captured_at, price FROM odds_snapshots
                WHERE league=? AND home_team=? AND away_team=? AND tip=? AND market=?
                  AND captured_at >= ?
                ORDER BY bookmaker, captured_at
                """, conn, params=(league, home, away, tip, market, cutoff)
            )
        if df.empty or current_price <= 1.01:
            return metrics
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce", utc=True, format="ISO8601")
        df = df.dropna(subset=["price", "captured_at"])
        moves = []
        for _bk, g in df.groupby("bookmaker"):
            if len(g) < 2:
                continue
            first = float(g.iloc[0]["price"]); last = float(g.iloc[-1]["price"])
            hours = max((g.iloc[-1]["captured_at"] - g.iloc[0]["captured_at"]).total_seconds() / 3600.0, 0.25)
            if first > 1.01 and last > 1.01:
                moves.append(((last / first) - 1.0, hours))
        if not moves:
            return metrics
        avg_move = float(np.mean([m[0] for m in moves]))
        avg_hours = float(np.mean([m[1] for m in moves]))
        velocity = avg_move / max(avg_hours, 0.25)
        same_dir = sum(1 for m, _h in moves if abs(m) >= STEAM_MIN_MOVE_PCT and np.sign(m) == np.sign(avg_move))
        steam_score = 0.0
        if same_dir >= STEAM_MIN_BOOKMAKERS:
            steam_score = float(min(max(avg_move / max(STEAM_MIN_MOVE_PCT * 3.0, 1e-9), -1.0), 1.0))
        predicted_closing = current_price * (1.0 + velocity * min(ODDS_VELOCITY_LOOKBACK_HOURS / 2.0, 8.0))
        predicted_closing = float(min(max(predicted_closing, 1.01), 20.0))
        expected_clv = (current_price / predicted_closing) - 1.0 if predicted_closing > 1.01 else 0.0
        metrics.update({
            "velocity": round(velocity, 5),
            "steam_score": round(steam_score, 4),
            "predicted_closing_odds": round(predicted_closing, 4),
            "expected_clv_pct": round(expected_clv, 5),
        })
        return metrics
    except Exception as e:
        log.debug("Odds velocity unavailable: %s", e)
        return metrics


def bayesian_update_probability(model_prob: float, market_prob: Optional[float], sharp_weight: float = 1.0) -> float:
    """Market prior + model evidence. Sharp market lines get a stronger prior."""
    if not BAYESIAN_UPDATE_ENABLED or market_prob is None or market_prob <= 0:
        return shrink_probability(model_prob)
    prior_strength = max(BAYES_MARKET_PRIOR_STRENGTH * max(sharp_weight, 0.25), 1.0)
    evidence_strength = max(BAYES_MODEL_EVIDENCE_STRENGTH, 1.0)
    alpha = market_prob * prior_strength + model_prob * evidence_strength
    beta = (1.0 - market_prob) * prior_strength + (1.0 - model_prob) * evidence_strength
    return float(min(max(alpha / (alpha + beta), 0.01), 0.99))


def ensemble_probability(model_prob: float, market_prob: Optional[float], bayes_prob: float, expected_clv_pct: Optional[float]) -> float:
    """Blend independent signals into one conservative probability."""
    if not ENSEMBLE_MODEL_ENABLED:
        return blended_probability(model_prob, market_prob)
    mkt = market_prob if market_prob is not None and market_prob > 0 else model_prob
    ecl_boost = 0.0
    if expected_clv_pct is not None:
        ecl_boost = min(max(expected_clv_pct, -0.03), 0.03) * 1.5
    raw = (ENSEMBLE_MODEL_WEIGHT * model_prob) + (ENSEMBLE_MARKET_WEIGHT * mkt) + (ENSEMBLE_BAYES_WEIGHT * bayes_prob) + (ENSEMBLE_ECL_WEIGHT * (mkt + ecl_boost))
    total_w = ENSEMBLE_MODEL_WEIGHT + ENSEMBLE_MARKET_WEIGHT + ENSEMBLE_BAYES_WEIGHT + ENSEMBLE_ECL_WEIGHT
    return shrink_probability(float(raw / max(total_w, 1e-9)))


def _parse_news_datetime(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        dt = pd.to_datetime(value, errors="coerce", utc=True, format="ISO8601")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def _load_injury_news_from_path(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        log.warning("INJURY_NEWS_SOURCE path neexistuje: %s", path)
        return []
    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw = raw.get("items", raw.get("news", raw.get("injuries", [])))
            return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path).fillna("").to_dict("records")
        log.warning("Nepodporovany INJURY_NEWS_SOURCE format: %s", path.suffix)
    except Exception as e:
        log.warning("Nacitavanie injury/news zdroja zlyhalo: %s", e)
    return []


async def refresh_injury_news_cache(session: aiohttp.ClientSession) -> int:
    """Load injury/news data once per run from a local JSON/CSV file or JSON URL.

    Expected item fields are flexible. Useful fields: league, team, opponent,
    player, status, headline, text, importance/risk, published_at, expires_at.
    The engine does not scrape random websites by default; you feed it a stable
    source/cache so bad HTML changes cannot break betting decisions.
    """
    global INJURY_NEWS_CACHE, INJURY_NEWS_CACHE_AT
    if not INJURY_NEWS_ENABLED:
        INJURY_NEWS_CACHE = []
        return 0
    if not INJURY_NEWS_SOURCE:
        log.warning("INJURY_NEWS_ENABLED=1, ale INJURY_NEWS_SOURCE nie je nastavene. Injury/news risk bude neutralny.")
        INJURY_NEWS_CACHE = []
        return 0
    now = now_utc()
    if INJURY_NEWS_CACHE_AT and (now - INJURY_NEWS_CACHE_AT).total_seconds() < INJURY_NEWS_CACHE_TTL_SECONDS:
        return len(INJURY_NEWS_CACHE)

    items: List[Dict[str, Any]] = []
    if INJURY_NEWS_SOURCE.lower().startswith(("http://", "https://")):
        try:
            async with session.get(INJURY_NEWS_SOURCE, timeout=HTTP_TIMEOUT) as resp:
                if resp.status != 200:
                    log.warning("Injury/news URL HTTP %s", resp.status)
                else:
                    raw = await resp.json(content_type=None)
                    if isinstance(raw, dict):
                        raw = raw.get("items", raw.get("news", raw.get("injuries", [])))
                    if isinstance(raw, list):
                        items = [x for x in raw if isinstance(x, dict)]
        except Exception as e:
            log.warning("Injury/news URL nacitanie zlyhalo: %s", e)
    else:
        p = Path(INJURY_NEWS_SOURCE)
        if not p.is_absolute():
            p = ROOT / p
        items = _load_injury_news_from_path(p)

    cutoff = now - timedelta(hours=INJURY_NEWS_MAX_AGE_HOURS)
    clean: List[Dict[str, Any]] = []
    for item in items:
        published = _parse_news_datetime(item.get("published_at") or item.get("created_at") or item.get("time"))
        expires = _parse_news_datetime(item.get("expires_at") or item.get("valid_until"))
        if published and published < cutoff:
            continue
        if expires and expires < now:
            continue
        clean.append(item)
    INJURY_NEWS_CACHE = clean
    INJURY_NEWS_CACHE_AT = now
    log.info("Injury/news cache loaded: %s relevant items", len(clean))
    return len(clean)


def _item_text(item: Dict[str, Any]) -> str:
    parts = [
        item.get("league", ""), item.get("team", ""), item.get("opponent", ""),
        item.get("player", ""), item.get("status", ""), item.get("headline", ""),
        item.get("title", ""), item.get("text", ""), item.get("summary", ""),
    ]
    return " ".join(str(x) for x in parts if x).lower()


def injury_news_risk_score(league: str, home: str, away: str, tip: str) -> float:
    """Return 0-1 risk from configured injury/news cache.

    A risk near 1 blocks the bet. A medium risk only reduces stake. The logic is
    conservative: negative news for the selected side increases risk, positive
    news for the opponent can also increase risk, and stale/unknown data is neutral.
    """
    if not INJURY_NEWS_ENABLED or not INJURY_NEWS_CACHE:
        return 0.0
    selected_team = home if tip == "1" else away if tip == "2" else ""
    opponent = away if selected_team == home else home if selected_team == away else ""
    risk = 0.0
    for item in INJURY_NEWS_CACHE:
        txt = _item_text(item)
        if league.lower() not in txt and str(item.get("league", "")).strip():
            continue
        team = str(item.get("team", "")).lower()
        mentions_selected = bool(selected_team and (selected_team.lower() in txt or normalize_name(selected_team) in normalize_name(txt) or normalize_name(team) == normalize_name(selected_team)))
        mentions_opponent = bool(opponent and (opponent.lower() in txt or normalize_name(opponent) in normalize_name(txt) or normalize_name(team) == normalize_name(opponent)))
        if not mentions_selected and not mentions_opponent:
            continue
        item_risk = safe_float(item.get("risk", item.get("importance", item.get("impact", 0.35))), 0.35)
        item_risk = float(min(max(item_risk, 0.05), 1.0))
        negative = any(k in txt for k in INJURY_NEGATIVE_KEYWORDS)
        positive = any(k in txt for k in INJURY_POSITIVE_KEYWORDS)
        if mentions_selected and negative:
            risk += item_risk
        elif mentions_selected and positive:
            risk -= item_risk * 0.35
        elif mentions_opponent and positive:
            risk += item_risk * 0.30
        elif mentions_opponent and negative:
            risk -= item_risk * 0.20
    return round(float(min(max(risk, 0.0), 1.0)), 4)


def live_betting_multiplier(is_live: bool, edge: float, price: float, qmetrics: Dict[str, Optional[float]]) -> Tuple[float, str]:
    """Safety layer for live mode.

    Live betting is allowed only when explicitly enabled and the market move data
    is good enough. Without a live score/event feed this stays very conservative.
    """
    if not is_live:
        return 1.0, "prematch"
    if not LIVE_BETTING_ENABLED:
        return 0.0, "live disabled"
    if price > LIVE_MAX_ODDS:
        return 0.0, f"live block: odds {price:.2f} > {LIVE_MAX_ODDS:.2f}"
    if edge < LIVE_MIN_EDGE:
        return 0.0, f"live block: edge {edge:.1%} < {LIVE_MIN_EDGE:.1%}"
    if LIVE_BLOCK_WITHOUT_VELOCITY and qmetrics.get("velocity") is None:
        return 0.0, "live block: no odds velocity"
    steam = qmetrics.get("steam_score")
    if LIVE_REQUIRE_GOOD_STEAM and (steam is None or steam < LIVE_MIN_STEAM_SCORE):
        return 0.0, f"live block: weak steam {steam}"
    return LIVE_STAKE_MULT, f"live stake x{LIVE_STAKE_MULT:.2f}"


def meta_confidence_score(
    tip: str,
    price: float,
    edge: float,
    market_edge: Optional[float],
    prob_model: float,
    prob_market: Optional[float],
    elo_diff: Optional[float],
    line_move: Optional[float],
) -> float:
    """Conservative 0-1 score for whether a value signal is credible."""
    if not META_LAYER_ENABLED:
        return 1.0
    score = 0.62
    score += min(max((edge - MIN_EDGE) / 0.12, -0.35), 0.35) * 0.22
    if market_edge is not None:
        score += min(max((market_edge - MIN_MARKET_EDGE) / 0.15, -0.35), 0.35) * 0.18
    if prob_market is not None:
        gap = abs(prob_model - prob_market)
        score -= min(max((gap - 0.10) / 0.20, 0.0), 1.0) * 0.12
    if line_move is not None:
        if line_move >= LINE_MOVE_GOOD_PCT:
            score += 0.08
        elif line_move <= LINE_MOVE_BAD_PCT:
            score -= 0.16
    if elo_diff is not None and tip in {"1", "2"}:
        # For home pick, positive ELO diff helps; for away pick, negative helps.
        aligned = elo_diff if tip == "1" else -elo_diff
        if aligned > 60:
            score += 0.06
        elif aligned < -40:
            score -= 0.10
    if tip == "X":
        score -= 0.05
    if price >= 4.5:
        score -= 0.06
    return round(float(min(max(score, 0.0), 1.0)), 4)


def meta_stake_multiplier(confidence: float) -> Tuple[float, str]:
    if not META_LAYER_ENABLED:
        return 1.0, "meta layer disabled"
    if confidence < META_MIN_CONFIDENCE:
        return 0.0, f"meta block: confidence {confidence:.1%} < {META_MIN_CONFIDENCE:.1%}"
    if confidence < META_REDUCE_BELOW:
        return META_REDUCED_STAKE_MULT, f"meta reduce x{META_REDUCED_STAKE_MULT:.2f}: confidence {confidence:.1%}"
    return 1.0, f"meta ok: confidence {confidence:.1%}"


def save_feature_store(bets: Sequence[BetCandidate]) -> int:
    if not FEATURE_STORE_ENABLED or not bets:
        return 0
    rows = []
    decision_at = now_utc().isoformat()
    for b in bets:
        rows.append((
            b.source_hash, decision_at, b.datum_iso, b.league, b.home_team, b.away_team,
            b.tip, b.market, b.bookmaker, b.kurz, b.prob_model, b.prob_market,
            b.prob_final, b.edge, b.market_edge, b.lh, b.la, b.elo_diff,
            b.line_move_pct, b.meta_confidence, b.ai_prob, b.score, APP_VERSION,
        ))
    with db_connect() as conn:
        before = conn.total_changes
        conn.executemany(
            """
            INSERT OR REPLACE INTO feature_store
            (source_hash, decision_at, datum_iso, league, home_team, away_team, tip, market,
             bookmaker, kurz, prob_model, prob_market, prob_final, edge, market_edge,
             lh, la, elo_diff, line_move_pct, meta_confidence, ai_prob, score, app_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return conn.total_changes - before

async def process_league(session: aiohttp.ClientSession, league: str, cfg: Dict[str, Any], model: Optional[Any], features: List[str]) -> List[BetCandidate]:
    hist_url = f"https://www.football-data.co.uk/mmz4281/{football_data_season_code()}/{cfg['csv']}.csv"
    df_hist = await fetch_csv(session, hist_url)
    if df_hist is None or len(df_hist) < 30:
        log.info("%s: malo historickych dat.", league)
        return []

    strengths = build_team_strengths(df_hist)
    elo_ratings = build_elo_ratings(df_hist)
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
            is_live = start < current
            if is_live:
                elapsed_min = (current - start).total_seconds() / 60.0
                if not LIVE_BETTING_ENABLED or elapsed_min < 0 or elapsed_min > LIVE_MAX_MATCH_MINUTES:
                    continue
            else:
                if not (current <= start <= horizon):
                    continue
            if SNAPSHOT_ODDS:
                save_match_odds_snapshot(league, match, home, away, start.isoformat())

            lh, la = expected_goals(home, away, strengths, float(cfg["ha"]))
            elo_diff = elo_diff_for_match(home, away, elo_ratings)
            probs = poisson_probs(lh, la, DIXON_COLES_RHO)
            consensus = market_consensus_probs_weighted(match.get("bookmakers", []), home_api, away_api)

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
                        sharp_weight = bookmaker_sharp_weight(bk_title)
                        bk_grade = bookmaker_grade(bk_title)
                        qmetrics = odds_velocity_metrics(league, home, away, tip, market_name, price)
                        bayes_prob = bayesian_update_probability(prob_model, prob_market, sharp_weight * bk_grade)
                        prob = ensemble_probability(prob_model, prob_market, bayes_prob, qmetrics.get("expected_clv_pct"))
                        edge = (prob * price) - 1.0
                        market_edge = ((prob_model - prob_market) if prob_market is not None else None)
                        # Sanity guards: reject weak disagreement, extreme model/market gaps,
                        # and candidates without enough bookmaker consensus. These reduce fake value.
                        if prob_market is None:
                            continue
                        if EXPECTED_CLV_ENABLED and qmetrics.get("expected_clv_pct") is not None and qmetrics["expected_clv_pct"] < EXPECTED_CLV_MIN:
                            log.info("Bet blocked by expected CLV: %.2f%% < %.2f%% | %s %s %s", qmetrics["expected_clv_pct"] * 100, EXPECTED_CLV_MIN * 100, league, f"{home} vs {away}", tip)
                            continue
                        if qmetrics.get("steam_score") is not None and qmetrics["steam_score"] <= -0.75 and qmetrics.get("expected_clv_pct", 0.0) < STEAM_BLOCK_BAD_MOVE_PCT:
                            log.info("Bet blocked by steam/velocity: steam %.2f expected CLV %.2f%% | %s %s %s", qmetrics["steam_score"], qmetrics.get("expected_clv_pct", 0.0) * 100, league, f"{home} vs {away}", tip)
                            continue
                        if market_edge is not None and market_edge < MIN_MARKET_EDGE:
                            continue
                        if abs(prob_model - prob_market) > MAX_MODEL_MARKET_GAP:
                            continue
                        if not is_reasonable_candidate(prob, price, edge):
                            continue
                        # Extra guards for noisy bet types. Draws and longshots can
                        # look attractive in a Poisson model while still being fragile.
                        if tip == "X" and (edge < DRAW_MIN_EDGE or price > DRAW_MAX_ODDS):
                            continue
                        if price > LONGSHOT_MAX_ODDS and (edge < LONGSHOT_MIN_EDGE or prob < LONGSHOT_MIN_PROB):
                            continue
                        line_move = line_movement_pct(league, home, away, tip, market_name, bk_title, price)
                        meta_conf = meta_confidence_score(tip, price, edge, market_edge, prob_model, prob_market, elo_diff, line_move)
                        meta_mult, meta_reason = meta_stake_multiplier(meta_conf)
                        if meta_mult <= 0:
                            log.info("Bet blocked by meta layer: %s | %s %s %s", meta_reason, league, f"{home} vs {away}", tip)
                            continue
                        injury_risk = injury_news_risk_score(league, home, away, tip)
                        if injury_risk >= INJURY_NEWS_RISK_BLOCK:
                            log.info("Bet blocked by injury/news risk: %.1f%% | %s %s %s", injury_risk * 100, league, f"{home} vs {away}", tip)
                            continue
                        injury_mult = INJURY_STAKE_REDUCE_MULT if injury_risk >= INJURY_NEWS_RISK_REDUCE else 1.0
                        live_mult, live_reason = live_betting_multiplier(is_live, edge, price, qmetrics)
                        if live_mult <= 0:
                            log.info("Bet blocked by live layer: %s | %s %s %s", live_reason, league, f"{home} vs {away}", tip)
                            continue
                        penalty = confidence_penalty(prob_model, prob_market, price, edge, bk_title)
                        ecl_bonus = max(qmetrics.get("expected_clv_pct") or 0.0, 0.0) * EXPECTED_CLV_SCORE_WEIGHT
                        steam_bonus = (qmetrics.get("steam_score") or 0.0) * STEAM_SCORE_WEIGHT
                        stake = max(0.0, kelly_stake(prob, price) * max(0.35, 1.0 - penalty / 10.0) * meta_mult * injury_mult * live_mult)
                        stake = round(min(stake, BANK * MAX_STAKE_PCT), 2)
                        if stake < MIN_STAKE:
                            continue
                        base_score = value_score(prob, price, edge, lh, la) - penalty + ((meta_conf - 0.60) * 4.0) + ecl_bonus + steam_bonus + ((bk_grade - 1.0) * 1.5)
                        ai_prob = None
                        score = base_score
                        if model is not None:
                            row = ai_candidate_row(
                                edge=edge,
                                market_edge=market_edge,
                                price=price,
                                lh=lh,
                                la=la,
                                prob_model=prob_model,
                                prob_market=prob_market,
                                prob_final=prob,
                                base_score=base_score,
                                tip=tip,
                                market_name=market_name,
                                elo_diff=elo_diff,
                                line_move_pct=line_move,
                                meta_confidence=meta_conf,
                            )
                            ai_prob = float(model.predict_proba(row)[0][1])
                            ai_edge = (ai_prob * price) - 1.0
                            if ai_prob < MIN_AI_PROB or ai_edge < MIN_AI_EDGE:
                                continue
                            score = round(base_score + (ai_edge * AI_SCORE_WEIGHT), 4)
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
                            elo_diff=round(elo_diff, 2),
                            line_move_pct=round(line_move, 5) if line_move is not None else None,
                            meta_confidence=round(meta_conf, 4),
                            sharp_weight=round(sharp_weight, 4),
                            bookmaker_grade=round(bk_grade, 4),
                            odds_velocity_pct_per_hour=qmetrics.get("velocity"),
                            steam_score=qmetrics.get("steam_score"),
                            predicted_closing_odds=qmetrics.get("predicted_closing_odds"),
                            expected_clv_pct=qmetrics.get("expected_clv_pct"),
                            bayes_prob=round(bayes_prob, 5),
                            ensemble_prob=round(prob, 5),
                            injury_news_risk=round(injury_risk, 4),
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


def recent_settled_profit(days: int = RISK_COOLDOWN_DAYS) -> Tuple[float, int]:
    """Return recent settled P/L and sample size for drawdown-aware staking."""
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with db_connect() as conn:
            df = pd.read_sql(
                """
                SELECT kurz, vklad, result, settled_at, created_at
                FROM bets
                WHERE result IN ('V','P')
                  AND COALESCE(settled_at, created_at) >= ?
                ORDER BY COALESCE(settled_at, created_at)
                """,
                conn,
                params=(cutoff,),
            )
        if df.empty:
            return 0.0, 0
        df["kurz"] = pd.to_numeric(df["kurz"], errors="coerce").fillna(0.0)
        df["vklad"] = pd.to_numeric(df["vklad"], errors="coerce").fillna(0.0)
        profit = np.where(df["result"] == "V", df["vklad"] * (df["kurz"] - 1.0), -df["vklad"])
        return float(profit.sum()), int(len(df))
    except Exception as e:
        log.warning("Risk cooldown check zlyhal: %s", e)
        return 0.0, 0


def current_risk_multiplier() -> Tuple[float, str]:
    """Reduce or pause new bets after a recent drawdown."""
    recent_profit, sample = recent_settled_profit(RISK_COOLDOWN_DAYS)
    if sample < 5:
        return 1.0, f"risk normal: malo nedavnych settled tipov ({sample})"
    loss_pct = -recent_profit / BANK if BANK > 0 and recent_profit < 0 else 0.0
    if loss_pct >= RISK_STOP_AFTER_LOSS_PCT:
        return 0.0, f"risk STOP: {RISK_COOLDOWN_DAYS}d P/L {recent_profit:.2f} EUR ({-loss_pct:.1%} banku)"
    if loss_pct >= RISK_REDUCE_AFTER_LOSS_PCT:
        mult = min(max(RISK_REDUCED_STAKE_MULT, 0.0), 1.0)
        return mult, f"risk reduced x{mult:.2f}: {RISK_COOLDOWN_DAYS}d P/L {recent_profit:.2f} EUR ({-loss_pct:.1%} banku)"
    return 1.0, f"risk normal: {RISK_COOLDOWN_DAYS}d P/L {recent_profit:.2f} EUR"


def scaled_bet(candidate: BetCandidate, stake: float) -> BetCandidate:
    """Return the same candidate with adjusted stake."""
    return BetCandidate(
        datum_iso=candidate.datum_iso,
        datum_display=candidate.datum_display,
        league=candidate.league,
        zapas=candidate.zapas,
        home_team=candidate.home_team,
        away_team=candidate.away_team,
        tip=candidate.tip,
        market=candidate.market,
        kurz=candidate.kurz,
        prob_model=candidate.prob_model,
        prob_market=candidate.prob_market,
        prob_final=candidate.prob_final,
        edge=candidate.edge,
        market_edge=candidate.market_edge,
        lh=candidate.lh,
        la=candidate.la,
        vklad=round(stake, 2),
        bookmaker=candidate.bookmaker,
        ai_prob=candidate.ai_prob,
        score=candidate.score,
        source_hash=candidate.source_hash,
        elo_diff=candidate.elo_diff,
        line_move_pct=candidate.line_move_pct,
        meta_confidence=candidate.meta_confidence,
    )


def historical_segment_stats(days: int = SEGMENT_LOOKBACK_DAYS) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Return quality metrics for segment keys like league=EPL or tip=X.

    The scanner uses this as a guardrail: segments with enough settled history
    and poor yield/CLV are either reduced or blocked. The minimum sample size is
    intentionally configurable because early databases can be noisy.
    """
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with db_connect() as conn:
            df = pd.read_sql(
                """
                SELECT league, market, bookmaker, tip, kurz, vklad, result, clv_pct, created_at
                FROM bets
                WHERE result IN ('V','P') AND COALESCE(created_at, '') >= ?
                """,
                conn,
                params=(cutoff,),
            )
        if df.empty:
            return {}
        df["kurz"] = pd.to_numeric(df["kurz"], errors="coerce").fillna(0.0)
        df["vklad"] = pd.to_numeric(df["vklad"], errors="coerce").fillna(0.0)
        df["clv_pct"] = pd.to_numeric(df.get("clv_pct", pd.Series(dtype=float)), errors="coerce")
        df["profit"] = np.where(df["result"] == "V", df["vklad"] * (df["kurz"] - 1.0), -df["vklad"])
        stats: Dict[Tuple[str, str], Dict[str, float]] = {}
        def add_stat(key_type: str, value: str, g: pd.DataFrame) -> None:
            turnover = float(g["vklad"].sum())
            profit = float(g["profit"].sum())
            clv = pd.to_numeric(g["clv_pct"], errors="coerce")
            stats[(key_type, str(value))] = {
                "settled": float(len(g)),
                "yield": profit / turnover if turnover else 0.0,
                "avg_clv": float(clv.mean()) if clv.notna().any() else np.nan,
                "clv_samples": float(clv.notna().sum()),
                "profit": profit,
            }

        for col in ["league", "market", "bookmaker", "tip"]:
            if col not in df.columns:
                continue
            values = df[col].fillna("unknown").replace("", "unknown")
            for value, g in df.groupby(values):
                add_stat(col, str(value), g)

        combo_cols = ["league", "market", "tip", "bookmaker"]
        if all(c in df.columns for c in combo_cols):
            combo_values = df[combo_cols].fillna("unknown").replace("", "unknown")
            combo_key = combo_values.astype(str).agg("|".join, axis=1)
            for value, g in df.groupby(combo_key):
                add_stat("combo", str(value), g)
        return stats
    except Exception as e:
        log.warning("Segment stats zlyhali: %s", e)
        return {}


def segment_quality_multiplier(candidate: BetCandidate, stats: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None) -> Tuple[float, str]:
    """Return 0 to block, <1 to reduce, 1 to keep based on historical segments."""
    stats = stats or historical_segment_stats()
    if not stats:
        return 1.0, "segment history unavailable"
    checks = [
        ("combo", f"{candidate.league}|{candidate.market}|{candidate.tip}|{candidate.bookmaker}", SEGMENT_COMBO_MIN_SETTLED),
        ("league", candidate.league, SEGMENT_MIN_SETTLED),
        ("market", candidate.market, SEGMENT_MIN_SETTLED),
        ("bookmaker", candidate.bookmaker, SEGMENT_MIN_SETTLED),
        ("tip", candidate.tip, SEGMENT_MIN_SETTLED),
    ]
    multiplier = 1.0
    reasons: List[str] = []
    for col, value, min_settled in checks:
        st = stats.get((col, str(value)))
        if not st or st.get("settled", 0) < min_settled:
            continue
        yld = float(st.get("yield", 0.0))
        avg_clv = st.get("avg_clv", np.nan)
        clv_samples_needed = max(SEGMENT_DISABLE_REQUIRE_CLV_SAMPLES, min_settled // 2)
        has_clv = not pd.isna(avg_clv) and st.get("clv_samples", 0) >= clv_samples_needed
        # Hard block only after enough samples. If CLV sample is small, yield alone
        # can still reduce stake, but it should not kill a segment prematurely.
        if yld <= SEGMENT_BLOCK_YIELD and (has_clv or st.get("settled", 0) >= min_settled * 2):
            reasons.append(f"BLOCK {col}={value} n={int(st.get('settled', 0))} yield={yld:.1%} clv={'n/a' if pd.isna(avg_clv) else f'{float(avg_clv):.2%}'}")
            return 0.0, "; ".join(reasons)
        if has_clv and float(avg_clv) <= SEGMENT_BLOCK_CLV:
            reasons.append(f"BLOCK {col}={value} n={int(st.get('settled', 0))} yield={yld:.1%} clv={float(avg_clv):.2%}")
            return 0.0, "; ".join(reasons)
        if yld <= SEGMENT_REDUCE_YIELD or (has_clv and float(avg_clv) <= SEGMENT_REDUCE_CLV):
            multiplier = min(multiplier, SEGMENT_REDUCED_STAKE_MULT)
            reasons.append(f"REDUCE {col}={value} n={int(st.get('settled', 0))} yield={yld:.1%} clv={'n/a' if pd.isna(avg_clv) else f'{float(avg_clv):.2%}'}")
    return multiplier, "; ".join(reasons) if reasons else "segment ok"


def apply_portfolio_limits(bets: Sequence[BetCandidate]) -> List[BetCandidate]:
    """Apply bankroll-level exposure controls after model ranking."""
    risk_mult, risk_reason = current_risk_multiplier()
    if risk_mult <= 0:
        log.warning("Portfolio paused by risk layer: %s", risk_reason)
        return []
    if risk_mult < 1:
        log.warning("Portfolio stakes reduced by risk layer: %s", risk_reason)
    else:
        log.info("Portfolio risk layer: %s", risk_reason)

    daily_cap = BANK * MAX_DAILY_EXPOSURE_PCT * risk_mult
    match_cap = BANK * MAX_MATCH_EXPOSURE_PCT * risk_mult
    league_cap = BANK * MAX_LEAGUE_EXPOSURE_PCT * risk_mult
    segment_stats = historical_segment_stats()
    accepted: List[BetCandidate] = []
    used_total = 0.0
    used_match: Dict[str, float] = {}
    used_league: Dict[str, float] = {}

    for b in sorted(bets, key=lambda x: (x.score, x.edge, x.prob_model), reverse=True):
        if len(accepted) >= MAX_BETS_PER_DAY:
            break
        seg_mult, seg_reason = segment_quality_multiplier(b, segment_stats)
        if seg_mult <= 0:
            log.info("Bet blocked by segment layer: %s | %s %s %s", seg_reason, b.league, b.zapas, b.tip)
            continue
        stake = round(b.vklad * risk_mult * seg_mult, 2)
        if stake < MIN_STAKE:
            continue
        match_key = f"{b.datum_iso}|{b.zapas}"
        if used_total + stake > daily_cap:
            continue
        if used_match.get(match_key, 0.0) + stake > match_cap:
            continue
        if used_league.get(b.league, 0.0) + stake > league_cap:
            continue
        accepted.append(scaled_bet(b, stake))
        used_total += stake
        used_match[match_key] = used_match.get(match_key, 0.0) + stake
        used_league[b.league] = used_league.get(b.league, 0.0) + stake
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
        df = pd.read_sql("SELECT kurz, vklad, result, created_at, settled_at FROM bets WHERE result IN ('V','P')", conn)
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



def settle_tip_from_score(tip: str, fthg: int, ftag: int, ftr: str) -> bool:
    """Return True when a backtested pick wins."""
    return (
        (tip == "1" and ftr == "H")
        or (tip == "X" and ftr == "D")
        or (tip == "2" and ftr == "A")
        or (tip == "Over 2.5" and (fthg + ftag) >= 3)
        or (tip == "Under 2.5" and (fthg + ftag) <= 2)
    )


def _prepare_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize football-data CSV rows for no-lookahead backtesting."""
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    if df is None or df.empty or any(c not in df.columns for c in required):
        return pd.DataFrame()
    out = df.dropna(subset=required).copy()
    out["match_date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out["FTHG"] = pd.to_numeric(out["FTHG"], errors="coerce")
    out["FTAG"] = pd.to_numeric(out["FTAG"], errors="coerce")
    out = out.dropna(subset=["match_date", "FTHG", "FTAG"])
    out["match_date_utc"] = pd.to_datetime(out["match_date"], utc=True)
    out["home_norm"] = out["HomeTeam"].astype(str)
    out["away_norm"] = out["AwayTeam"].astype(str)
    return out


def _market_consensus_from_snapshot(group: pd.DataFrame, market: str, tip: str) -> Optional[float]:
    """No-vig consensus using only odds visible in the same captured snapshot."""
    probs: List[float] = []
    if market == "h2h":
        needed = {"1", "X", "2"}
    elif market == "totals":
        needed = {"Over 2.5", "Under 2.5"}
    else:
        return None
    for _bk, g in group[group["market"] == market].groupby("bookmaker"):
        prices = {str(r["tip"]): safe_float(r["price"], 0.0) for _, r in g.iterrows()}
        if not needed.issubset(prices):
            continue
        nv = no_vig_probabilities(prices)
        if tip in nv:
            probs.append(nv[tip])
    return float(np.mean(probs)) if len(probs) >= MIN_CONSENSUS_BOOKMAKERS else None


def _closing_price_from_snapshots(
    snapshots: pd.DataFrame,
    league: str,
    home: str,
    away: str,
    tip: str,
    market: str,
    bookmaker: str,
    kickoff: pd.Timestamp,
) -> Optional[float]:
    """Estimate closing odds from the last snapshot before kickoff."""
    if snapshots.empty:
        return None
    lower = kickoff - pd.Timedelta(hours=CLV_LOOKBACK_HOURS)
    mask = (
        (snapshots["league"] == league)
        & (snapshots["home_team"] == home)
        & (snapshots["away_team"] == away)
        & (snapshots["tip"] == tip)
        & (snapshots["market"] == market)
        & (snapshots["captured_at_dt"] >= lower)
        & (snapshots["captured_at_dt"] <= kickoff)
    )
    same_book = snapshots[mask & (snapshots["bookmaker"] == bookmaker)].sort_values("captured_at_dt")
    if not same_book.empty:
        return safe_float(same_book.iloc[-1]["price"], 0.0)
    market_rows = snapshots[mask].sort_values("captured_at_dt")
    if market_rows.empty:
        return None
    last_by_book = market_rows.groupby("bookmaker").tail(1)
    closing = float(pd.to_numeric(last_by_book["price"], errors="coerce").mean())
    return closing if closing > 1.01 else None


def _simulate_portfolio_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Apply the same portfolio limits to historical simulated candidates."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["captured_at", "score", "edge"], ascending=[True, False, False])
    accepted: List[pd.Series] = []
    for decision_time, batch in df.groupby("captured_at", sort=True):
        used_total = 0.0
        used_match: Dict[str, float] = {}
        used_league: Dict[str, float] = {}
        daily_cap = BANK * MAX_DAILY_EXPOSURE_PCT
        match_cap = BANK * MAX_MATCH_EXPOSURE_PCT
        league_cap = BANK * MAX_LEAGUE_EXPOSURE_PCT
        count = 0
        deduped = batch.sort_values(["score", "edge", "price"], ascending=False).drop_duplicates(["commence_time", "league", "home_team", "away_team", "market", "tip"])
        for _, r in deduped.iterrows():
            if count >= MAX_BETS_PER_DAY:
                break
            stake = safe_float(r["stake"], 0.0)
            match_key = f"{r['commence_time']}|{r['home_team']}|{r['away_team']}"
            league = str(r["league"])
            if used_total + stake > daily_cap:
                continue
            if used_match.get(match_key, 0.0) + stake > match_cap:
                continue
            if used_league.get(league, 0.0) + stake > league_cap:
                continue
            accepted.append(r)
            used_total += stake
            used_match[match_key] = used_match.get(match_key, 0.0) + stake
            used_league[league] = used_league.get(league, 0.0) + stake
            count += 1
    return pd.DataFrame(accepted) if accepted else pd.DataFrame()


async def real_backtest_report(session: aiohttp.ClientSession, days: int = BACKTEST_DAYS) -> str:
    """No-lookahead backtest from stored odds_snapshots.

    This does not backtest already saved bets. It rebuilds team strengths only
    from matches played before each odds snapshot, evaluates the exact odds that
    were stored at that time, applies the same value filters and portfolio caps,
    then settles against later final scores from football-data.co.uk.

    Important limitation: it can only test periods where odds_snapshots were
    actually collected before kickoff. Historical bookmaker odds are not
    downloaded retroactively by this script.
    """
    init_db()
    with db_connect() as conn:
        snapshots = pd.read_sql("SELECT * FROM odds_snapshots ORDER BY captured_at", conn)
    if snapshots.empty:
        return (
            "REAL BACKTEST: odds_snapshots je prázdna.\n"
            "Najprv nechaj live scanner pravidelne ukladať snapshoty pred zápasmi "
            "alebo importuj historické snapshoty do tabuľky odds_snapshots."
        )

    snapshots["captured_at_dt"] = pd.to_datetime(snapshots["captured_at"], errors="coerce", utc=True)
    snapshots["commence_dt"] = pd.to_datetime(snapshots["commence_time"], errors="coerce", utc=True)
    snapshots["price"] = pd.to_numeric(snapshots["price"], errors="coerce")
    snapshots = snapshots.dropna(subset=["captured_at_dt", "commence_dt", "price"])
    snapshots = snapshots[(snapshots["price"] > 1.01) & (snapshots["captured_at_dt"] < snapshots["commence_dt"])]
    if days > 0:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        snapshots = snapshots[snapshots["captured_at_dt"] >= cutoff]
    if ACTIVE_LEAGUES is not None:
        snapshots = snapshots[snapshots["league"].isin(ACTIVE_LEAGUES)]
    if snapshots.empty:
        return f"REAL BACKTEST: za posledných {days} dní nie sú použiteľné pre-match snapshoty."

    rows: List[Dict[str, Any]] = []
    skipped_no_history = 0
    skipped_unsettled = 0

    for league, cfg in LIGY.items():
        if ACTIVE_LEAGUES is not None and league not in ACTIVE_LEAGUES:
            continue
        league_snaps = snapshots[snapshots["league"] == league].copy()
        if league_snaps.empty:
            continue

        hist_url = f"https://www.football-data.co.uk/mmz4281/{football_data_season_code()}/{cfg['csv']}.csv"
        hist_raw = await fetch_csv(session, hist_url)
        hist = _prepare_results_df(hist_raw)
        if hist.empty:
            continue

        for key, group in league_snaps.groupby(["captured_at", "commence_time", "home_team", "away_team"], sort=True):
            captured_at, commence_time, home, away = key
            decision_dt = pd.to_datetime(captured_at, utc=True, errors="coerce")
            kickoff_dt = pd.to_datetime(commence_time, utc=True, errors="coerce")
            if pd.isna(decision_dt) or pd.isna(kickoff_dt):
                continue

            past = hist[hist["match_date_utc"] < decision_dt.normalize()].copy()
            if len(past) < 30:
                skipped_no_history += 1
                continue
            teams = set(past["HomeTeam"].astype(str)) | set(past["AwayTeam"].astype(str))
            home_match = match_team(str(home), teams)
            away_match = match_team(str(away), teams)
            if not home_match or not away_match:
                skipped_no_history += 1
                continue

            settled = hist[(hist["HomeTeam"] == home_match) & (hist["AwayTeam"] == away_match)].copy()
            if not settled.empty:
                settled["date_diff"] = settled["match_date_utc"].apply(lambda d: abs((d.date() - kickoff_dt.date()).days))
                settled = settled[settled["date_diff"] <= 1].sort_values("date_diff")
            if settled.empty:
                skipped_unsettled += 1
                continue
            result_row = settled.iloc[0]
            fthg, ftag, ftr = int(result_row["FTHG"]), int(result_row["FTAG"]), str(result_row["FTR"])

            strengths = build_team_strengths(past)
            if home_match not in strengths or away_match not in strengths:
                skipped_no_history += 1
                continue
            lh, la = expected_goals(home_match, away_match, strengths, float(cfg["ha"]))
            probs = poisson_probs(lh, la, DIXON_COLES_RHO)

            for _, snap in group.iterrows():
                tip = str(snap["tip"])
                market = str(snap["market"])
                price = safe_float(snap["price"], 0.0)
                bookmaker = str(snap["bookmaker"])
                if tip not in probs or price <= 1.01:
                    continue
                prob_model = probs[tip]
                prob_market = _market_consensus_from_snapshot(group, market, tip)
                if prob_market is None:
                    continue
                prob_final = blended_probability(prob_model, prob_market)
                edge = (prob_final * price) - 1.0
                market_edge = prob_model - prob_market
                if market_edge < MIN_MARKET_EDGE:
                    continue
                if abs(prob_model - prob_market) > MAX_MODEL_MARKET_GAP:
                    continue
                if not is_reasonable_candidate(prob_final, price, edge):
                    continue
                penalty = confidence_penalty(prob_model, prob_market, price, edge, bookmaker)
                stake = max(0.0, kelly_stake(prob_final, price) * max(0.35, 1.0 - penalty / 10.0))
                stake = round(min(stake, BANK * MAX_STAKE_PCT), 2)
                if stake < MIN_STAKE:
                    continue
                base_score = value_score(prob_final, price, edge, lh, la) - penalty
                won = settle_tip_from_score(tip, fthg, ftag, ftr)
                profit = stake * (price - 1.0) if won else -stake
                closing = _closing_price_from_snapshots(snapshots, league, str(home), str(away), tip, market, bookmaker, kickoff_dt)
                clv_pct = ((price / closing) - 1.0) if closing and closing > 1.01 else np.nan
                rows.append({
                    "captured_at": str(captured_at),
                    "captured_at_dt": decision_dt,
                    "commence_time": str(commence_time),
                    "league": league,
                    "home_team": home_match,
                    "away_team": away_match,
                    "market": market,
                    "tip": tip,
                    "price": round(price, 4),
                    "stake": stake,
                    "profit": round(profit, 4),
                    "won": int(won),
                    "prob_model": prob_model,
                    "prob_market": prob_market,
                    "prob_final": prob_final,
                    "edge": edge,
                    "market_edge": market_edge,
                    "lh": lh,
                    "la": la,
                    "bookmaker": bookmaker,
                    "score": base_score,
                    "closing_price": closing,
                    "clv_pct": clv_pct,
                })

    selected = _simulate_portfolio_from_rows(rows)
    if selected.empty:
        return (
            f"REAL BACKTEST: žiadne tipy neprešli filtrami za posledných {days} dní.\n"
            f"Kandidáti pred portfolio limitmi: {len(rows)} | preskočené bez histórie: {skipped_no_history} | bez výsledku: {skipped_unsettled}."
        )

    selected = selected.sort_values("captured_at_dt")
    turnover = float(selected["stake"].sum())
    profit = float(selected["profit"].sum())
    total = int(len(selected))
    wins = int(selected["won"].sum())
    hit_rate = wins / total if total else 0.0
    yield_pct = profit / turnover if turnover else 0.0
    equity = selected["profit"].cumsum()
    max_dd = float((equity.cummax() - equity).max()) if not equity.empty else 0.0
    avg_edge = float(selected["edge"].mean())
    avg_clv = float(pd.to_numeric(selected["clv_pct"], errors="coerce").mean()) if selected["clv_pct"].notna().any() else np.nan
    clv_win = float((selected["clv_pct"].dropna() > 0).mean()) if selected["clv_pct"].notna().any() else np.nan

    export_path = EXPORT_DIR / f"real_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    selected.to_csv(export_path, index=False)

    lines = [
        f"REAL BACKTEST ENGINE v{APP_VERSION}",
        f"Obdobie: posledných {days} dní | Zdroj: odds_snapshots + football-data výsledky",
        f"Snapshot rows: {len(snapshots)} | Kandidáti pred portfolio limitmi: {len(rows)} | Vybrané tipy: {total}",
        f"Turnover {turnover:.2f} EUR | P/L {profit:.2f} EUR | Yield {yield_pct:.1%} | Win rate {hit_rate:.1%} | Max DD {max_dd:.2f} EUR",
        f"Avg edge {avg_edge:.1%} | Avg CLV {'n/a' if pd.isna(avg_clv) else f'{avg_clv:.2%}'} | CLV+ {'n/a' if pd.isna(clv_win) else f'{clv_win:.1%}'}",
        f"Calibration Brier: {float(((selected['won'] - selected['prob_final']) ** 2).mean()):.4f} | Avg P {float(selected['prob_final'].mean()):.1%} | Real WR {hit_rate:.1%}",
        f"Preskočené bez histórie: {skipped_no_history} | bez výsledku: {skipped_unsettled}",
        f"CSV export: {export_path}",
        "",
        "Podľa ligy:",
    ]
    by_league = selected.groupby("league").agg(bets=("profit", "size"), stake=("stake", "sum"), profit=("profit", "sum"), win_rate=("won", "mean"), avg_clv=("clv_pct", "mean")).sort_values("profit", ascending=False)
    for league, r in by_league.iterrows():
        y = r["profit"] / r["stake"] if r["stake"] else 0.0
        clv = "n/a" if pd.isna(r["avg_clv"]) else f"{r['avg_clv']:.2%}"
        lines.append(f"- {league}: {int(r['bets'])} tipov | P/L {r['profit']:.2f} EUR | Yield {y:.1%} | WR {r['win_rate']:.1%} | CLV {clv}")
    lines.append("")
    lines.append("Podľa marketu:")
    by_market = selected.groupby("market").agg(bets=("profit", "size"), stake=("stake", "sum"), profit=("profit", "sum"), win_rate=("won", "mean"), avg_clv=("clv_pct", "mean")).sort_values("profit", ascending=False)
    for market, r in by_market.iterrows():
        y = r["profit"] / r["stake"] if r["stake"] else 0.0
        clv = "n/a" if pd.isna(r["avg_clv"]) else f"{r['avg_clv']:.2%}"
        lines.append(f"- {market}: {int(r['bets'])} tipov | P/L {r['profit']:.2f} EUR | Yield {y:.1%} | WR {r['win_rate']:.1%} | CLV {clv}")
    lines.append("")
    lines.append("Top 10 bookmakrov podľa P/L:")
    by_book = selected.groupby("bookmaker").agg(bets=("profit", "size"), stake=("stake", "sum"), profit=("profit", "sum"), avg_clv=("clv_pct", "mean")).sort_values("profit", ascending=False).head(10)
    for book, r in by_book.iterrows():
        y = r["profit"] / r["stake"] if r["stake"] else 0.0
        clv = "n/a" if pd.isna(r["avg_clv"]) else f"{r['avg_clv']:.2%}"
        lines.append(f"- {book}: {int(r['bets'])} tipov | P/L {r['profit']:.2f} EUR | Yield {y:.1%} | CLV {clv}")
    lines.append("")
    lines.append("Poznámka: toto je reálny no-lookahead backtest len pre časy, v ktorých už máš uložené odds_snapshots.")
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



def walk_forward_validation_report(df: pd.DataFrame) -> List[str]:
    """Chronological out-of-sample audit using settled bets only.

    The report does not retune old picks. For every calendar month it treats all
    previous settled bets as the training history and the current month as the
    forward test window. This is a practical sanity check for overfitting and
    probability calibration drift.
    """
    if df.empty or "result" not in df.columns:
        return ["Walk-forward: bez dát."]
    settled = df[df["result"].isin(["V", "P"])].copy()
    if settled.empty:
        return ["Walk-forward: zatiaľ bez vyhodnotených tipov."]
    date_col = "datum_iso" if "datum_iso" in settled.columns else "created_at"
    settled["wf_date"] = pd.to_datetime(settled[date_col], errors="coerce", utc=True)
    if settled["wf_date"].isna().all() and "created_at" in settled.columns:
        settled["wf_date"] = pd.to_datetime(settled["created_at"], errors="coerce", utc=True)
    settled = settled.dropna(subset=["wf_date"]).sort_values("wf_date")
    if len(settled) < WALK_FORWARD_MIN_TRAIN_ROWS + WALK_FORWARD_MIN_TEST_ROWS:
        return [
            "Walk-forward: málo dát na seriózny audit "
            f"({len(settled)} settled, potreba aspoň {WALK_FORWARD_MIN_TRAIN_ROWS + WALK_FORWARD_MIN_TEST_ROWS})."
        ]
    settled["prob_final"] = pd.to_numeric(_safe_col(settled, "prob_final", np.nan), errors="coerce")
    settled["kurz"] = pd.to_numeric(_safe_col(settled, "kurz", 0.0), errors="coerce").fillna(0.0)
    settled["vklad"] = pd.to_numeric(_safe_col(settled, "vklad", 0.0), errors="coerce").fillna(0.0)
    settled["clv_pct"] = pd.to_numeric(_safe_col(settled, "clv_pct", np.nan), errors="coerce")
    settled["win"] = (settled["result"] == "V").astype(int)
    settled["profit"] = np.where(settled["win"] == 1, settled["vklad"] * (settled["kurz"] - 1.0), -settled["vklad"])
    settled["period"] = settled["wf_date"].dt.to_period("M").astype(str)

    rows: List[Dict[str, Any]] = []
    for period, test in settled.groupby("period", sort=True):
        train = settled[settled["wf_date"] < test["wf_date"].min()]
        if len(train) < WALK_FORWARD_MIN_TRAIN_ROWS or len(test) < WALK_FORWARD_MIN_TEST_ROWS:
            continue
        turnover = float(test["vklad"].sum())
        profit = float(test["profit"].sum())
        rows.append({
            "period": period,
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "yield": profit / turnover if turnover else np.nan,
            "profit": profit,
            "win_rate": float(test["win"].mean()),
            "avg_prob": float(test["prob_final"].mean()) if test["prob_final"].notna().any() else np.nan,
            "brier": float(((test["win"] - test["prob_final"]) ** 2).mean()) if test["prob_final"].notna().any() else np.nan,
            "avg_clv": float(test["clv_pct"].mean()) if test["clv_pct"].notna().any() else np.nan,
        })
    if not rows:
        return ["Walk-forward: dáta existujú, ale mesačné okná sú príliš malé."]
    res = pd.DataFrame(rows)
    lines = ["Walk-forward out-of-sample audit"]
    avg_brier_txt = "n/a" if res["brier"].isna().all() else f"{res['brier'].mean():.4f}"
    lines.append(
        f"- Okná: {len(res)} | Priemerný test yield {res['yield'].mean():.1%} | "
        f"Median yield {res['yield'].median():.1%} | Avg Brier {avg_brier_txt}"
    )
    tail = res.tail(8)
    for _, r in tail.iterrows():
        clv = "n/a" if pd.isna(r["avg_clv"]) else f"{r['avg_clv']:.2%}"
        brier = "n/a" if pd.isna(r["brier"]) else f"{r['brier']:.4f}"
        lines.append(
            f"- {r['period']}: train {int(r['train_n'])} | test {int(r['test_n'])} | "
            f"Yield {r['yield']:.1%} | WR {r['win_rate']:.1%} | AvgP {r['avg_prob']:.1%} | Brier {brier} | CLV {clv}"
        )
    return lines

def _format_combo_segments(df: pd.DataFrame, min_settled: int = SEGMENT_MIN_SETTLED, top_n: int = 12) -> List[str]:
    """Audit league/market/tip/bookmaker combinations; useful for auto-disabling weak pockets."""
    cols = ["league", "market", "tip", "bookmaker"]
    if df.empty or any(c not in df.columns for c in cols):
        return ["Segment combo audit: bez dát."]
    settled_df = df[df["is_settled"]].copy()
    if settled_df.empty:
        return ["Segment combo audit: bez vyhodnotených tipov."]
    rows = []
    for key, g in settled_df.groupby([settled_df[c].fillna("unknown").replace("", "unknown") for c in cols]):
        if len(g) < min_settled:
            continue
        turnover = float(g["vklad"].sum())
        profit = float(g["profit"].sum())
        clv = pd.to_numeric(g.get("clv_pct", pd.Series(dtype=float)), errors="coerce")
        rows.append({
            "segment": " | ".join(str(x) for x in key),
            "settled": len(g),
            "profit": profit,
            "yield": profit / turnover if turnover else np.nan,
            "avg_clv": float(clv.mean()) if clv.notna().any() else np.nan,
            "clv_samples": int(clv.notna().sum()),
        })
    if not rows:
        return [f"Segment combo audit: málo dát pre min_settled={min_settled}."]
    res = pd.DataFrame(rows)
    weak = res[(res["yield"] <= SEGMENT_REDUCE_YIELD) | ((res["clv_samples"] >= 5) & (res["avg_clv"] <= SEGMENT_REDUCE_CLV))]
    if weak.empty:
        return ["Segment combo audit: žiadne kombinácie zatiaľ nevyzerajú slabé."]
    weak = weak.sort_values(["yield", "avg_clv", "settled"], ascending=[True, True, False]).head(top_n)
    lines = ["Slabé kombinácie liga | market | tip | bookmaker"]
    for _, r in weak.iterrows():
        clv = "n/a" if pd.isna(r["avg_clv"]) else f"{r['avg_clv']:.2%}"
        lines.append(f"- {r['segment']}: settled {int(r['settled'])} | Yield {r['yield']:.1%} | P/L {r['profit']:.2f} EUR | Avg CLV {clv}")
    return lines


def _format_weak_segments(df: pd.DataFrame, min_settled: int = 8) -> List[str]:
    """Highlight segments that look dangerous enough to tighten or disable."""
    if df.empty:
        return ["Rizikové segmenty: bez dát."]
    lines = ["Rizikové segmenty na audit"]
    found = 0
    for group_col in ["league", "market", "bookmaker", "tip"]:
        if group_col not in df.columns:
            continue
        for key, g in df.groupby(df[group_col].fillna("unknown").replace("", "unknown")):
            settled = g[g["is_settled"]]
            if len(settled) < min_settled:
                continue
            turnover = float(settled["vklad"].sum())
            profit = float(settled["profit"].sum())
            yld = profit / turnover if turnover else 0.0
            clv = pd.to_numeric(g.get("clv_pct", pd.Series(dtype=float)), errors="coerce")
            avg_clv = float(clv.mean()) if clv.notna().any() else np.nan
            if yld < -0.08 or (not pd.isna(avg_clv) and avg_clv < -0.015):
                found += 1
                clv_txt = "n/a" if pd.isna(avg_clv) else f"{avg_clv:.2%}"
                lines.append(
                    f"- {group_col}={key}: settled {len(settled)} | Yield {yld:.1%} | "
                    f"P/L {profit:.2f} EUR | Avg CLV {clv_txt}"
                )
    if found == 0:
        lines.append("- Žiadny segment zatiaľ nespĺňa hranicu na tvrdý zásah.")
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
    lines.extend(walk_forward_validation_report(df))
    lines.append("")
    lines.extend(_format_group_table(df, "league", "Výkon podľa ligy"))
    lines.append("")
    lines.extend(_format_group_table(df, "market", "Výkon podľa marketu"))
    lines.append("")
    lines.extend(_format_group_table(df, "bookmaker", "Výkon podľa bookmakera"))
    lines.append("")
    lines.extend(_format_group_table(df, "tip", "Výkon podľa tipu"))
    lines.append("")
    lines.extend(_format_combo_segments(df))
    lines.append("")
    lines.extend(_format_weak_segments(df))
    lines.append("")
    lines.append("Interpretácia:")
    lines.append("- Priorita je kladné CLV na veľkej vzorke; krátkodobý profit môže byť len variance.")
    lines.append("- Segmenty s negatívnym CLV a slabým yieldom sú kandidáti na vypnutie alebo vyšší MIN_EDGE.")
    lines.append("- Ak calibration bucket dlhodobo nadhodnocuje real win rate, zníž MARKET_BLEND_WEIGHT alebo sprísni MIN_MARKET_EDGE.")
    lines.append("- Walk-forward audit ber ako hlavný signál stability: dobrý historický yield bez forward stability je riziko prefitovania.")
    return "\n".join(lines)


# =============================== REPORTING ===============================

def export_csv(bets: Sequence[BetCandidate]) -> Optional[Path]:
    if not bets:
        return None
    path = EXPORT_DIR / f"value_bets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields = [
        "datum_display", "league", "zapas", "tip", "market", "kurz", "prob_model",
        "prob_market", "prob_final", "edge", "market_edge", "implied_prob", "fair_odds", "ev_eur", "risk_level",
        "lh", "la", "vklad", "bookmaker", "ai_prob", "elo_diff", "line_move_pct", "meta_confidence",
        "sharp_weight", "bookmaker_grade", "odds_velocity_pct_per_hour", "steam_score", "predicted_closing_odds",
        "expected_clv_pct", "bayes_prob", "ensemble_prob", "injury_news_risk", "score", "explanation"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for b in bets:
            row = {k: getattr(b, k) for k in fields if k != "explanation"}
            row["explanation"] = explain_candidate(b)
            writer.writerow(row)
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


def explain_candidate(b: BetCandidate) -> str:
    """Human-readable reason why the candidate survived the filters."""
    parts = [
        f"value edge {b.edge:.1%} vs implied {b.implied_prob:.1%}",
        f"model {b.prob_model:.1%} vs market {((b.prob_market or 0.0)):.1%}",
        f"market edge {((b.market_edge or 0.0)):.1%}",
        f"fair odds {b.fair_odds} vs offered {b.kurz}",
    ]
    if b.tip == "X":
        parts.append(f"draw guard passed: edge>={DRAW_MIN_EDGE:.1%}, odds<={DRAW_MAX_ODDS}")
    if b.kurz > LONGSHOT_MAX_ODDS:
        parts.append(f"longshot guard passed: edge>={LONGSHOT_MIN_EDGE:.1%}, prob>={LONGSHOT_MIN_PROB:.1%}")
    if b.elo_diff is not None:
        parts.append(f"ELO diff {b.elo_diff:+.0f}")
    if b.line_move_pct is not None:
        parts.append(f"line movement {b.line_move_pct:+.2%}")
    if b.meta_confidence is not None:
        parts.append(f"meta confidence {b.meta_confidence:.1%}")
    if b.expected_clv_pct is not None:
        parts.append(f"expected CLV {b.expected_clv_pct:+.2%}; predicted close {b.predicted_closing_odds or 0:.3f}")
    if b.steam_score is not None:
        parts.append(f"steam score {b.steam_score:+.2f}; velocity {(b.odds_velocity_pct_per_hour or 0):+.2%}/h")
    if b.bookmaker_grade is not None:
        parts.append(f"bookmaker grade {b.bookmaker_grade:.2f}; sharp weight {(b.sharp_weight or 1):.2f}")
    if b.bayes_prob is not None:
        parts.append(f"Bayes probability {b.bayes_prob:.1%}; ensemble {b.ensemble_prob or b.prob_final:.1%}")
    if b.injury_news_risk is not None and b.injury_news_risk > 0:
        parts.append(f"injury/news risk {b.injury_news_risk:.1%}")
    if b.ai_prob is not None:
        parts.append(f"AI probability {b.ai_prob:.1%}")
    parts.append(f"stake capped by bankroll/segment/risk limits at {b.vklad:.2f} EUR")
    return "; ".join(parts)

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
            f"| Meta {(b.meta_confidence or 0):.1%} | ELO {(b.elo_diff or 0):+.0f} "
            f"| ECL {((b.expected_clv_pct or 0)*100):+.2f}% | Steam {(b.steam_score or 0):+.2f} "
            f"| Grade {(b.bookmaker_grade or 1):.2f} | {b.bookmaker}{ai}\n"
            f"    Prečo prešiel: {explain_candidate(b)}"
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
    if HEALTH_ONLY:
        print(runtime_health_report())
        return
    if BANKROLL_SIM_ONLY:
        print(bankroll_monte_carlo_report(BANKROLL_SIM_PATHS, BANKROLL_SIM_BETS, BANKROLL_SIM_STAKE_PCT))
        return
    if ANALYTICS_ONLY:
        print(analytics_report(ANALYTICS_DAYS))
        return

    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if COLLECT_ODDS_ONLY:
            print(await collect_odds_snapshots(session, loop_once=not COLLECTOR_LOOP, interval_seconds=COLLECTOR_INTERVAL_SECONDS))
            return
        if BACKTEST_ONLY:
            print(await real_backtest_report(session, BACKTEST_DAYS))
            return
        settled = await settle_results(session)
        if SETTLE_ONLY:
            print(f"Vyhodnotene tipy: {settled}\n{performance_summary()}")
            return
        if INJURY_NEWS_ENABLED:
            await refresh_injury_news_cache(session)
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
    features_saved = 0 if DRY_RUN else save_feature_store(bets)
    if features_saved:
        log.info("Feature store rows saved: %s", features_saved)
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
    parser.add_argument("--backtest", action="store_true", help="Spusti realny no-lookahead backtest zo stored odds_snapshots")
    parser.add_argument("--analytics", action="store_true", help="Zobraz audit modelu: CLV, calibration, ligy, markety, bookmakeri")
    parser.add_argument("--analytics-days", type=int, default=ANALYTICS_DAYS, help="Pocet dni pre analytics report")
    parser.add_argument("--backtest-days", type=int, default=BACKTEST_DAYS, help="Pocet dni pre realny backtest zo snapshotov")
    parser.add_argument("--league", action="append", choices=sorted(LIGY.keys()), help="Obmedz sken na ligu; mozes zadat viackrat")
    parser.add_argument("--min-edge", type=float, help="Docasne prepis MIN_EDGE, napr. 0.06")
    parser.add_argument("--no-email", action="store_true", help="Neposielaj email report")
    parser.add_argument("--top", type=int, help="Max pocet tipov v reporte")
    parser.add_argument("--bank", type=float, help="Docasne prepis bank")
    parser.add_argument("--live", action="store_true", help="Zapni konzervativny live betting layer pre tento run")
    parser.add_argument("--injury-news", action="store_true", help="Zapni injury/news risk layer pre tento run")
    parser.add_argument("--injury-source", type=str, help="JSON/CSV cesta alebo JSON URL pre injury/news data")
    parser.add_argument("--health", action="store_true", help="Zobraz runtime health/prod readiness report")
    parser.add_argument("--collect-odds", action="store_true", help="Len uloz odds snapshots bez vytvarania tipov")
    parser.add_argument("--collector-loop", action="store_true", help="Pri --collect-odds bez v loop-e pre 24/7 zber odds")
    parser.add_argument("--collector-interval", type=int, default=COLLECTOR_INTERVAL_SECONDS, help="Interval odds collectora v sekundach")
    parser.add_argument("--bankroll-sim", action="store_true", help="Spusti Monte Carlo bankroll simulator")
    parser.add_argument("--sim-paths", type=int, default=BANKROLL_SIM_PATHS, help="Pocet Monte Carlo ciest")
    parser.add_argument("--sim-bets", type=int, default=BANKROLL_SIM_BETS, help="Pocet betov v jednej Monte Carlo ceste")
    parser.add_argument("--sim-stake-pct", type=float, default=BANKROLL_SIM_STAKE_PCT, help="Stake percento banku pre Monte Carlo")
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global ACTIVE_LEAGUES, DRY_RUN, SETTLE_ONLY, NO_EMAIL, BACKTEST_ONLY, ANALYTICS_ONLY, HEALTH_ONLY, COLLECT_ODDS_ONLY, COLLECTOR_LOOP, BANKROLL_SIM_ONLY, COLLECTOR_INTERVAL_SECONDS, BANKROLL_SIM_PATHS, BANKROLL_SIM_BETS, BANKROLL_SIM_STAKE_PCT, ANALYTICS_DAYS, BACKTEST_DAYS, MIN_EDGE, TOP_N_REPORT, BANK, LIVE_BETTING_ENABLED, INJURY_NEWS_ENABLED, INJURY_NEWS_SOURCE
    ACTIVE_LEAGUES = set(args.league) if args.league else None
    DRY_RUN = bool(args.dry_run)
    SETTLE_ONLY = bool(args.settle_only)
    NO_EMAIL = bool(args.no_email or args.dry_run)
    BACKTEST_ONLY = bool(args.backtest)
    ANALYTICS_ONLY = bool(args.analytics)
    HEALTH_ONLY = bool(getattr(args, "health", False))
    COLLECT_ODDS_ONLY = bool(getattr(args, "collect_odds", False))
    COLLECTOR_LOOP = bool(getattr(args, "collector_loop", False))
    BANKROLL_SIM_ONLY = bool(getattr(args, "bankroll_sim", False))
    COLLECTOR_INTERVAL_SECONDS = int(getattr(args, "collector_interval", COLLECTOR_INTERVAL_SECONDS))
    BANKROLL_SIM_PATHS = int(getattr(args, "sim_paths", BANKROLL_SIM_PATHS))
    BANKROLL_SIM_BETS = int(getattr(args, "sim_bets", BANKROLL_SIM_BETS))
    BANKROLL_SIM_STAKE_PCT = float(getattr(args, "sim_stake_pct", BANKROLL_SIM_STAKE_PCT))
    ANALYTICS_DAYS = int(args.analytics_days)
    BACKTEST_DAYS = int(args.backtest_days)
    if args.min_edge is not None:
        MIN_EDGE = float(args.min_edge)
    if args.top is not None:
        TOP_N_REPORT = int(args.top)
    if args.bank is not None:
        BANK = float(args.bank)
    if getattr(args, "live", False):
        LIVE_BETTING_ENABLED = 1
    if getattr(args, "injury_news", False):
        INJURY_NEWS_ENABLED = 1
    if getattr(args, "injury_source", None):
        INJURY_NEWS_SOURCE = str(args.injury_source)


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
