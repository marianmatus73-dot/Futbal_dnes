# V16.00 merge base — tip-optimized main
# Built from V15.21 + pipeline registry + professional tip selection layer.
# Behaviour-compatible env flags; stronger gates, stakes, diversification, tip-first report.
#
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sqlite3
import smtplib
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from v16_00_master_integration import run_master_cycle  # noqa: F401
from v16_main_hook import run_v16_main_hook
from v16_16_profile_pipeline import run_pipeline as run_v16_profile_pipeline
from v16_production_inputs import build_production_inputs
from v16_cycle_store import save_cycle
from v16_dashboard import build_dashboard
from v16_alerting import evaluate_alerts

from core.multisport_learning_v2.manager import MultisportLearningV2Manager
from core.config import Settings
from core.reporting import print_report
from core.sport_settlement import ensure_settlement_columns
from core.audit_summary import audit_block_summary
from core.performance_summary import performance_report
from core.bet_converter import bet_to_tip_dict
from core.bankroll import bankroll_summary
from core.pro_tipper import (
    build_pro_tip,
    filter_value_tips,
    sort_tips,
    save_tip_audit_log,
    format_pro_report,
    rejected_tips,
    format_rejected_report,
)
from core.top_tips import select_top_tips, select_telegram_tips
from core.learning_model import retrain_from_results
from core.consensus_engine import ConsensusInput, build_consensus
from core.football_learning import run_football_learning
from core.football_meta_ai_v14 import run_football_meta_ai_v14
from core.football_data_collector_v14 import run_football_data_collector_v14
from core.football_maintenance_v14 import run_football_maintenance_v14
from core.football_postmatch_dataset_v14 import rebuild_football_postmatch_dataset_v14
from core.football_dataset_v15 import rebuild_football_dataset_v15
from core.football_evaluation_dashboard_v15 import run_football_evaluation_dashboard_v15
from core.football_feature_importance_v15 import run_feature_importance_v15
from core.football_insights_v15 import run_football_insights_v15
from core.football_ai_health_v15 import run_ai_health_report_v15
from core.football_learning_progress_v15 import run_learning_progress_v15
from core.football_learning_progress_v15_4 import run_learning_progress_v15_4
from core.football_data_quality_booster_v15_5 import run_data_quality_booster_v15_5
from core.football_closing_odds_collector_v15_6 import run_closing_odds_collector_v15_6
from core.football_xg_collector_v15_7 import run_xg_collector_v15_7
from core.football_data_readiness_v15_8 import run_data_readiness_v15_8
from core.football_data_capture_engine_v15_9_1 import run_data_capture_v15_9_1
from core.football_closing_odds_capture_v15_10 import run_closing_odds_capture_v15_10
from core.football_closing_line_resolver_v15_11 import run_closing_line_resolver_v15_11
from core.football_closing_odds_storage_clv_v15_12 import run_closing_storage_clv_v15_12
from core.football_closing_odds_backfill_v15_13 import run_closing_backfill_v15_13
from core.football_closing_snapshot_matcher_v15_14 import run_snapshot_matcher_v15_14
from core.football_real_closing_snapshot_extractor_v15_15 import (
    run_real_closing_snapshot_extractor_v15_15,
)
from core.football_match_id_resolver_v15_16 import run_match_id_resolver_v15_16
from core.football_universal_match_key_builder_v15_17 import (
    run_universal_match_key_builder_v15_17,
)
from core.football_universal_join_executor_v15_18 import run_universal_join_executor_v15_18
from core.football_closing_odds_database_join_engine_v15_19 import (
    run_closing_odds_database_join_engine_v15_19,
)
from core.football_snapshot_schema_extractor_v15_20 import (
    run_snapshot_schema_extractor_v15_20,
)
from core.football_closing_odds_writer_v15_21 import run_closing_odds_writer_v15_21
from core.football_result_learning import run_football_result_learning
from core.football_settlement import settle_football_bets
from core.football_trainer import ensure_feature_history_table
from core.football_xg import FootballXGDatabase
from core.football_elo import FootballEloDatabase
from core.football_team_form import FootballFormDatabase
from core.football_league_calibration import (
    FootballLeagueCalibrationDatabase,
    rebuild_football_league_calibrations,
)
from core.football_team_xg_v14 import FootballTeamXGV14Database
from core.football_team_elo_v14 import FootballTeamEloV14Database

from sports.football import FootballModule
from sports.tennis import TennisModule
from sports.basketball import BasketballModule
from sports.hockey import HockeyModule
from sports.baseball import BaseballModule
from sports.mma import MMAModule
from sports.nfl import NFLModule

try:
    from core.sport_quant import init_sport_db
except Exception:  # pragma: no cover
    init_sport_db = None


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("multisport-main")

LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Bratislava"))

SPORT_MODULES = [
    FootballModule(),
    TennisModule(),
    BasketballModule(),
    HockeyModule(),
    BaseballModule(),
    MMAModule(),
    NFLModule(),
]

HISTORY_EXPORTS = {
    "sport_bets": "exports/history_sport_bets.csv",
    "sport_bookmaker_stats": "exports/history_bookmaker_stats.csv",
    "sport_elo_ratings": "exports/history_elo_ratings.csv",
    "football_feature_history": "exports/history_football_features.csv",
    "football_xg_ratings": "exports/history_football_xg_ratings.csv",
    "football_xg_history": "exports/history_football_xg_matches.csv",
    "football_elo_ratings": "exports/history_football_elo_ratings.csv",
    "football_elo_history": "exports/history_football_elo_matches.csv",
    "football_team_form": "exports/history_football_team_form.csv",
    "football_form_history": "exports/history_football_form_matches.csv",
    "football_result_learning_state": "exports/history_football_result_learning_state.csv",
    "football_settlement_audit": "exports/history_football_settlement_audit.csv",
    "football_league_calibration": "exports/history_football_league_calibration.csv",
    "football_team_xg_v14": "exports/history_football_team_xg_v14.csv",
    "football_team_elo_v14": "exports/history_football_team_elo_v14.csv",
    "football_team_elo_v14_history": "exports/history_football_team_elo_v14_matches.csv",
    "football_market_snapshots_v14": "exports/history_football_market_snapshots_v14.csv",
    "football_xg_history_v14": "exports/history_football_xg_history_v14.csv",
    "football_postmatch_dataset_v14": "exports/history_football_postmatch_dataset_v14.csv",
    "football_dataset_v15": "exports/history_football_dataset_v15.csv",
    "football_explainability_v15": "exports/history_football_explainability_v15.csv",
}

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Tip policy (env-tunable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TipPolicy:
    top_limit: int = 5
    min_publish: int = 1
    min_edge: float = 0.03
    odds_min: float = 1.50
    odds_max: float = 3.20
    min_confidence: int = 70
    min_signals: int = 2
    max_per_match: int = 1
    max_per_sport: int = 2
    max_per_league: int = 2
    telegram_min_confidence: int = 80
    kelly_fraction: float = 0.25
    stake_min_u: float = 0.50
    stake_max_u: float = 2.50
    bankroll_units: float = 100.0
    unit_size: float = 1.0
    blocked_leagues: tuple[str, ...] = ()
    require_value_filter: bool = True
    email_only_if_publishable: bool = True

    @classmethod
    def from_env(cls) -> "TipPolicy":
        blocked_raw = os.getenv("BLOCKED_LEAGUES", "")
        blocked = tuple(
            x.strip().lower()
            for x in blocked_raw.split(",")
            if x.strip()
        )
        return cls(
            top_limit=int(os.getenv("TOP_TIPS_LIMIT", "5")),
            min_publish=int(os.getenv("MIN_TIPS_TO_PUBLISH", "1")),
            min_edge=float(os.getenv("MIN_EDGE", "0.03")),
            odds_min=float(os.getenv("ODDS_MIN", "1.50")),
            odds_max=float(os.getenv("ODDS_MAX", "3.20")),
            min_confidence=int(os.getenv("MIN_TIP_CONFIDENCE", "70")),
            min_signals=int(os.getenv("MIN_TIP_SIGNALS", "2")),
            max_per_match=int(os.getenv("MAX_TIPS_PER_MATCH", "1")),
            max_per_sport=int(os.getenv("MAX_TIPS_PER_SPORT", "2")),
            max_per_league=int(os.getenv("MAX_TIPS_PER_LEAGUE", "2")),
            telegram_min_confidence=int(os.getenv("TELEGRAM_MIN_CONFIDENCE", "80")),
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
            stake_min_u=float(os.getenv("STAKE_MIN_U", "0.50")),
            stake_max_u=float(os.getenv("STAKE_MAX_U", "2.50")),
            bankroll_units=float(os.getenv("BANKROLL_UNITS", "100")),
            unit_size=float(os.getenv("UNIT_SIZE", "1.0")),
            blocked_leagues=blocked,
            require_value_filter=os.getenv("REQUIRE_VALUE_FILTER", "1") == "1",
            email_only_if_publishable=os.getenv("EMAIL_ONLY_IF_PUBLISHABLE", "1") == "1",
        )


# ---------------------------------------------------------------------------
# Context + pipeline primitives
# ---------------------------------------------------------------------------


@dataclass
class StepError:
    name: str
    message: str
    critical: bool = False


@dataclass
class RejectedTip:
    sport: str
    match: str
    pick: str
    odds: float | None
    reason: str


@dataclass
class ScoredTip:
    """Normalized tip card used by selection / stakes / report."""

    tip: Any
    sport: str
    league: str
    match: str
    pick: str
    odds: float
    bookmaker: str
    model_probability: float
    market_probability: float
    edge: float
    confidence: int
    score: float
    signals: int
    signal_names: str
    reason: str
    stake_u: float = 0.0
    kelly_raw: float = 0.0

    @property
    def match_key(self) -> str:
        return f"{self.sport.lower()}::{self.match.lower().strip()}"

    @property
    def league_key(self) -> str:
        return f"{self.sport.lower()}::{self.league.lower().strip()}"

    @property
    def pick_key(self) -> str:
        return (
            f"{self.match_key}::{self.pick.lower().strip()}::"
            f"{round(self.odds, 3)}"
        )


@dataclass
class TipCard:
    """Final selection package for reporting / email / audit."""

    all_tips: list[Any]
    value_tips: list[Any]
    gated: list[ScoredTip]
    selected: list[ScoredTip]
    rejected: list[RejectedTip]
    telegram: list[ScoredTip]
    publishable: bool
    total_stake_u: float
    policy: TipPolicy


@dataclass
class RunContext:
    settings: Settings
    args: argparse.Namespace
    module_outputs: list[dict] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    errors: list[StepError] = field(default_factory=list)
    tip_card: TipCard | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(LOCAL_TZ))

    def set(self, key: str, value: Any) -> Any:
        self.results[key] = value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self.results.get(key, default)

    def dataset_training_ready(self) -> int:
        ds = self.get("football_dataset_v15")
        if ds is None:
            return 0
        return int(getattr(ds, "training_ready", 0) or 0)

    def dataset_with_closing(self) -> int:
        ds = self.get("football_dataset_v15")
        if ds is None:
            return 0
        return int(getattr(ds, "with_closing", 0) or 0)

    def feature_importance(self) -> dict:
        value = self.get("football_feature_importance") or {}
        return value if isinstance(value, dict) else {}

    def record_error(
        self,
        name: str,
        exc: BaseException,
        *,
        critical: bool = False,
    ) -> None:
        self.errors.append(StepError(name=name, message=str(exc), critical=critical))

    @property
    def critical_failed(self) -> bool:
        return any(e.critical for e in self.errors)

    @property
    def is_full_scan(self) -> bool:
        return not (self.args.dry_run or self.args.analytics or self.args.backtest)


StepFn = Callable[[RunContext], Any]
AsyncStepFn = Callable[[RunContext], Awaitable[Any]]


@dataclass(frozen=True)
class PipelineStep:
    name: str
    env_flag: str
    run: StepFn | AsyncStepFn
    enabled_default: str = "1"
    only_full_scan: bool = True
    critical: bool = False
    store_as: str | None = None


def env_enabled(flag: str, default: str = "1") -> bool:
    return os.getenv(flag, default) == "1"


def should_run_step(step: PipelineStep, ctx: RunContext) -> bool:
    if step.only_full_scan and not ctx.is_full_scan:
        return False
    return env_enabled(step.env_flag, step.enabled_default)


async def run_pipeline(steps: Iterable[PipelineStep], ctx: RunContext) -> None:
    for step in steps:
        if not should_run_step(step, ctx):
            log.debug("Skipping step %s", step.name)
            continue

        started = time.perf_counter()
        try:
            result = step.run(ctx)
            if asyncio.iscoroutine(result):
                result = await result
            key = step.store_as or step.name
            ctx.set(key, result)
            elapsed = time.perf_counter() - started
            log.info("Step OK: %s (%.2fs)", step.name, elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            log.exception("Step FAILED: %s (%.2fs)", step.name, elapsed)
            ctx.record_error(step.name, exc, critical=step.critical)
            ctx.set(step.store_as or step.name, None)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def db_path(settings: Settings) -> Path:
    return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))


def db_connect(settings: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path(settings)), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def safe_table_name(table: str) -> str:
    if table not in HISTORY_EXPORTS:
        raise ValueError(f"Unsupported table: {table}")
    if not _SAFE_IDENT.match(table):
        raise ValueError(f"Unsafe table name: {table}")
    return table


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]


def quote_ident(name: str) -> str:
    if not _SAFE_IDENT.match(name):
        raise ValueError(f"Unsafe identifier: {name}")
    return f'"{name}"'


def db_execute_with_retry(
    fn: Callable[[], Any],
    *,
    retries: int = 5,
    delay: float = 0.05,
) -> Any:
    last: BaseException | None = None
    for attempt in range(retries):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            last = exc
            if "locked" not in str(exc).lower() or attempt == retries - 1:
                raise
            time.sleep(delay * (2 ** attempt))
    if last:
        raise last
    return None


def import_csv_to_table(settings: Settings, table: str, csv_file: str) -> int:
    table = safe_table_name(table)
    path = Path(csv_file)
    if not path.exists():
        return 0

    def _do() -> int:
        with db_connect(settings) as conn:
            if not table_exists(conn, table):
                return 0

            db_cols = set(get_table_columns(conn, table))

            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return 0

                raw_columns = list(reader.fieldnames)
                columns = [
                    c for c in raw_columns if c in db_cols and _SAFE_IDENT.match(c)
                ]
                skipped = [c for c in raw_columns if c not in db_cols]
                if skipped:
                    log.debug(
                        "CSV %s: ignoring columns not in %s: %s",
                        path.name,
                        table,
                        ", ".join(skipped[:12]) + ("..." if len(skipped) > 12 else ""),
                    )
                if not columns:
                    log.warning(
                        "CSV %s has no overlapping columns with %s",
                        path.name,
                        table,
                    )
                    return 0

                rows = list(reader)
                if not rows:
                    return 0

                placeholders = ",".join(["?"] * len(columns))
                col_sql = ",".join(quote_ident(c) for c in columns)
                sql = (
                    f"INSERT OR IGNORE INTO {quote_ident(table)} "
                    f"({col_sql}) VALUES ({placeholders})"
                )
                values = [[row.get(col, "") for col in columns] for row in rows]
                before = conn.total_changes
                conn.executemany(sql, values)
                conn.commit()
                return conn.total_changes - before

    return int(db_execute_with_retry(_do) or 0)


def export_table_to_csv(settings: Settings, table: str, csv_file: str) -> int:
    table = safe_table_name(table)
    path = Path(csv_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _do() -> tuple[list[str], list[tuple]]:
        with db_connect(settings) as conn:
            if not table_exists(conn, table):
                return [], []
            cursor = conn.execute(f"SELECT * FROM {quote_ident(table)}")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return columns, rows

    columns, rows = db_execute_with_retry(_do)
    if not columns:
        return 0

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    return len(rows)


def table_count(settings: Settings, table: str) -> int | None:
    try:
        if table in HISTORY_EXPORTS:
            table = safe_table_name(table)
        elif not _SAFE_IDENT.match(table):
            return None
    except ValueError:
        if not _SAFE_IDENT.match(table):
            return None
    try:
        with db_connect(settings) as conn:
            if not table_exists(conn, table):
                return None
            row = conn.execute(
                f"SELECT COUNT(*) FROM {quote_ident(table)}"
            ).fetchone()
            return int(row[0]) if row else 0
    except Exception as exc:
        log.debug("table_count(%s) failed: %s", table, exc)
        return None


# ---------------------------------------------------------------------------
# Learning history bootstrap
# ---------------------------------------------------------------------------


def init_football_v13_learning_tables(settings: Settings) -> None:
    ensure_feature_history_table(db_path(settings))
    FootballXGDatabase(settings).init_db()
    FootballEloDatabase(settings).init_db()
    FootballFormDatabase(settings).init_db()
    FootballLeagueCalibrationDatabase(settings).init_db()
    FootballTeamXGV14Database(settings).init_db()
    FootballTeamEloV14Database(settings).init_db()


def ensure_football_settlement_columns(settings: Settings) -> None:
    database = db_path(settings)
    database.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(database)) as conn:
        exists = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name='sport_bets'
            """
        ).fetchone()
        if exists is None:
            return

        existing_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(sport_bets)").fetchall()
        }
        required_columns = {
            "home_goals": "INTEGER",
            "away_goals": "INTEGER",
            "final_score": "TEXT",
            "settled_at": "TEXT",
            "settlement_source": "TEXT",
            "external_event_id": "TEXT",
        }
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns and _SAFE_IDENT.match(column_name):
                conn.execute(
                    f"ALTER TABLE sport_bets ADD COLUMN {column_name} {column_type}"
                )
        conn.commit()


def restore_learning_history(settings: Settings) -> None:
    if init_sport_db is not None:
        try:
            init_sport_db(settings)
        except Exception as e:
            log.warning("Could not init sport learning tables: %s", e)

    try:
        ensure_settlement_columns(settings)
    except Exception as e:
        log.warning("Could not ensure settlement columns: %s", e)

    try:
        ensure_football_settlement_columns(settings)
    except Exception as e:
        log.warning("Could not ensure Football v13 settlement columns: %s", e)

    try:
        init_football_v13_learning_tables(settings)
    except Exception as e:
        log.warning("Could not init Football v13 learning tables: %s", e)

    total = 0
    for table, csv_file in HISTORY_EXPORTS.items():
        try:
            imported = import_csv_to_table(settings, table, csv_file)
            total += imported
            if imported:
                log.info("Imported %s rows into %s", imported, table)
        except Exception as e:
            log.warning("History import failed for %s: %s", table, e)

    log.info("Learning history restore finished. Imported rows: %s", total)


def save_learning_history(settings: Settings) -> None:
    total = 0
    for table, csv_file in HISTORY_EXPORTS.items():
        try:
            exported = export_table_to_csv(settings, table, csv_file)
            total += exported
            log.info("Exported %s rows from %s", exported, table)
        except Exception as e:
            log.warning("History export failed for %s: %s", table, e)
    log.info("Learning history export finished. Exported rows: %s", total)


# ---------------------------------------------------------------------------
# CLI / email / sport runners
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multisport betting engine")
    parser.add_argument(
        "--sport",
        choices=["all"] + sorted([m.name for m in SPORT_MODULES]),
        default=os.getenv("SPORT_MODE", "all"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analytics", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--no-email", action="store_true")
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=int(os.getenv("BACKTEST_DAYS", "180")),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("SPORT_CONCURRENCY", "3")),
    )
    parser.add_argument(
        "--full-maintenance",
        action="store_true",
        help="Reserved flag for full maintenance path (default full scan already runs it).",
    )
    return parser.parse_args()


def send_multisport_email(body: str, *, subject_prefix: str | None = None) -> bool:
    gmail_user = os.getenv("GMAIL_USER", "").strip()
    gmail_password = os.getenv("GMAIL_PASSWORD", "").strip()
    gmail_receiver = os.getenv("GMAIL_RECEIVER", gmail_user).strip()

    if not gmail_user or not gmail_password or not gmail_receiver:
        log.info("Email credentials missing - multisport email skipped.")
        return False

    prefix = subject_prefix or "Top Pro Betting Tips"
    subject = (
        f"{prefix} - {datetime.now(LOCAL_TZ).strftime('%d.%m.%Y %H:%M')}"
    )

    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = gmail_receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
        log.info("Multisport email report sent to %s", gmail_receiver)
        return True
    except Exception as e:
        log.warning("Multisport email failed: %s", e)
        return False


async def run_sport_module(
    sport: Any,
    settings: Settings,
    args: argparse.Namespace,
) -> dict:
    started = datetime.now(LOCAL_TZ)
    try:
        log.info("Running sport module: %s", sport.name)

        if args.analytics:
            result = await sport.analytics(settings)
        elif args.backtest:
            result = await sport.backtest(settings, days=args.backtest_days)
        else:
            result = await sport.scan(settings)

        duration = (datetime.now(LOCAL_TZ) - started).total_seconds()
        return {
            "sport": sport.name,
            "ok": True,
            "duration_sec": duration,
            "result": result,
            "error": None,
        }
    except Exception as e:
        duration = (datetime.now(LOCAL_TZ) - started).total_seconds()
        log.exception("Sport module failed: %s", sport.name)
        return {
            "sport": sport.name,
            "ok": False,
            "duration_sec": duration,
            "result": None,
            "error": str(e),
        }


def to_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def tip_get(tip: Any, key: str, default: Any = None) -> Any:
    if isinstance(tip, dict):
        return tip.get(key, default)
    return getattr(tip, key, default)


def tip_confidence(tip: Any) -> int:
    for key in ("confidence", "confidence_score", "conf"):
        raw = tip_get(tip, key)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
            if 0 <= value <= 1:
                return int(round(value * 100))
            return int(round(value))
        except (TypeError, ValueError):
            continue
    edge = to_float_or_none(tip_get(tip, "edge")) or to_float_or_none(
        tip_get(tip, "raw_edge")
    )
    if edge is None:
        return 0
    return int(max(0, min(99, 55 + edge * 400)))


def tip_score(tip: Any, edge: float, confidence: int, signals: int) -> float:
    explicit = to_float_or_none(tip_get(tip, "score")) or to_float_or_none(
        tip_get(tip, "model_score")
    )
    if explicit is not None:
        base = explicit
    else:
        base = edge * 100.0 + confidence * 0.35 + signals * 2.0
    return float(base)


# ---------------------------------------------------------------------------
# Professional tip selection layer
# ---------------------------------------------------------------------------


def extract_raw_pro_tips(module_outputs: list[dict]) -> list[Any]:
    """Convert module outputs → ProTip objects (pre-filter)."""
    raw_tips: list[Any] = []

    for item in module_outputs:
        result = item.get("result")
        if not result:
            continue

        if isinstance(result, dict):
            candidates = (
                result.get("tips")
                or result.get("picks")
                or result.get("bets")
                or []
            )
        elif isinstance(result, list):
            candidates = result
        else:
            candidates = getattr(result, "bets", []) or []

        for candidate in candidates:
            tip = bet_to_tip_dict(candidate, fallback_sport=item["sport"])
            if not tip:
                continue

            try:
                odds = to_float_or_none(tip.get("odds"))
                if odds is None or odds <= 1:
                    continue

                market_probability = to_float_or_none(tip.get("market_probability"))
                if market_probability is None:
                    market_probability = 1.0 / odds

                elo_p = to_float_or_none(tip.get("elo_probability"))
                xg_p = to_float_or_none(tip.get("xg_probability"))
                form_p = to_float_or_none(tip.get("form_probability"))

                consensus = build_consensus(
                    ConsensusInput(
                        sport=tip.get("sport", item["sport"]),
                        league=tip.get("league", "Unknown"),
                        match=tip.get("match") or tip.get("event") or "Unknown",
                        pick=tip.get("pick") or tip.get("selection") or "Unknown",
                        odds=odds,
                        elo_probability=elo_p,
                        xg_probability=xg_p,
                        form_probability=form_p,
                        market_probability=market_probability,
                        injury_penalty=to_float_or_none(tip.get("injury_penalty"))
                        or 0.0,
                        news_penalty=to_float_or_none(tip.get("news_penalty")) or 0.0,
                    )
                )

                reason_parts: list[str] = []
                if tip.get("reason"):
                    reason_parts.append(str(tip.get("reason")))
                if consensus.reason:
                    reason_parts.append(consensus.reason)

                signal_names = []
                if elo_p is not None:
                    signal_names.append("ELO")
                if xg_p is not None:
                    signal_names.append("xG")
                if form_p is not None:
                    signal_names.append("FORM")
                signal_names.append("MKT")

                pro_tip = build_pro_tip(
                    sport=consensus.sport,
                    league=consensus.league,
                    match=consensus.match,
                    pick=consensus.pick,
                    odds=consensus.odds,
                    model_probability=consensus.model_probability,
                    bookmaker=tip.get("bookmaker", ""),
                    reason=" | ".join(reason_parts),
                    raw_edge=to_float_or_none(tip.get("raw_edge")),
                    model_score=tip.get("model_score"),
                )

                try:
                    if hasattr(pro_tip, "__dict__"):
                        pro_tip.__dict__["_signals"] = len(signal_names)
                        pro_tip.__dict__["_signal_names"] = "+".join(signal_names)
                        pro_tip.__dict__["_market_probability"] = market_probability
                        pro_tip.__dict__["_model_probability"] = (
                            consensus.model_probability
                        )
                    elif isinstance(pro_tip, dict):
                        pro_tip["_signals"] = len(signal_names)
                        pro_tip["_signal_names"] = "+".join(signal_names)
                        pro_tip["_market_probability"] = market_probability
                        pro_tip["_model_probability"] = consensus.model_probability
                except Exception:
                    pass

                raw_tips.append(pro_tip)
            except Exception as e:
                log.warning("Could not convert consensus tip to ProTip: %s", e)

    return sort_tips(raw_tips)


def extract_pro_tips(module_outputs: list[dict]) -> tuple[list, list]:
    """Backward-compatible wrapper used by older report paths."""
    all_tips = extract_raw_pro_tips(module_outputs)
    value_tips = filter_value_tips(all_tips)
    log.info("Extracted %s raw pro tips before value filter", len(all_tips))
    log.info("Value tips after filter: %s", len(value_tips))
    return all_tips, sort_tips(value_tips)


def _normalize_scored(
    tip: Any,
    *,
    rejected: list[RejectedTip],
    policy: TipPolicy,
) -> ScoredTip | None:
    sport = str(tip_get(tip, "sport", "Unknown") or "Unknown")
    league = str(tip_get(tip, "league", "Unknown") or "Unknown")
    match = str(tip_get(tip, "match") or tip_get(tip, "event") or "Unknown")
    pick = str(tip_get(tip, "pick") or tip_get(tip, "selection") or "Unknown")
    bookmaker = str(tip_get(tip, "bookmaker", "") or "")
    odds = to_float_or_none(tip_get(tip, "odds"))

    def reject(reason: str) -> None:
        rejected.append(
            RejectedTip(
                sport=sport,
                match=match,
                pick=pick,
                odds=odds,
                reason=reason,
            )
        )

    if odds is None or odds <= 1:
        reject("invalid odds")
        return None

    if league.lower().strip() in policy.blocked_leagues:
        reject(f"blocked league: {league}")
        return None

    if not (policy.odds_min <= odds <= policy.odds_max):
        reject(f"odds {odds:.2f} outside {policy.odds_min:.2f}-{policy.odds_max:.2f}")
        return None

    model_p = (
        to_float_or_none(tip_get(tip, "_model_probability"))
        or to_float_or_none(tip_get(tip, "model_probability"))
        or to_float_or_none(tip_get(tip, "probability"))
    )
    market_p = (
        to_float_or_none(tip_get(tip, "_market_probability"))
        or to_float_or_none(tip_get(tip, "market_probability"))
    )
    if market_p is None:
        market_p = 1.0 / odds
    if model_p is None:
        model_p = market_p

    edge = to_float_or_none(tip_get(tip, "edge"))
    if edge is None:
        raw_edge = to_float_or_none(tip_get(tip, "raw_edge"))
        edge = raw_edge if raw_edge is not None else (model_p - market_p)

    if edge < policy.min_edge:
        reject(f"edge {edge:.3f} < min {policy.min_edge:.3f}")
        return None

    confidence = tip_confidence(tip)
    if confidence < policy.min_confidence:
        reject(f"confidence {confidence} < min {policy.min_confidence}")
        return None

    signals = int(tip_get(tip, "_signals") or 0)
    signal_names = str(tip_get(tip, "_signal_names") or "MKT")
    if tip_get(tip, "_signals") is None:
        signals = 1
        signal_names = "MKT?"
    elif signals < policy.min_signals:
        reject(f"signals {signals} < min {policy.min_signals}")
        return None

    score = tip_score(tip, edge=edge, confidence=confidence, signals=signals)
    reason = str(tip_get(tip, "reason", "") or "")

    return ScoredTip(
        tip=tip,
        sport=sport,
        league=league,
        match=match,
        pick=pick,
        odds=float(odds),
        bookmaker=bookmaker,
        model_probability=float(model_p),
        market_probability=float(market_p),
        edge=float(edge),
        confidence=int(confidence),
        score=float(score),
        signals=int(signals),
        signal_names=signal_names,
        reason=reason,
    )


def dedupe_best_odds(scored: list[ScoredTip]) -> list[ScoredTip]:
    """Keep best odds for identical sport/match/pick."""
    best: dict[str, ScoredTip] = {}
    for tip in scored:
        key = f"{tip.match_key}::{tip.pick.lower().strip()}"
        prev = best.get(key)
        if prev is None or tip.odds > prev.odds or (
            tip.odds == prev.odds and tip.score > prev.score
        ):
            best[key] = tip
    return list(best.values())


def select_diversified_tips(
    scored: list[ScoredTip],
    *,
    policy: TipPolicy,
    rejected: list[RejectedTip],
) -> list[ScoredTip]:
    ordered = sorted(scored, key=lambda t: (t.score, t.edge, t.confidence), reverse=True)
    picked: list[ScoredTip] = []
    match_count: dict[str, int] = {}
    sport_count: dict[str, int] = {}
    league_count: dict[str, int] = {}

    for tip in ordered:
        if match_count.get(tip.match_key, 0) >= policy.max_per_match:
            rejected.append(
                RejectedTip(
                    tip.sport,
                    tip.match,
                    tip.pick,
                    tip.odds,
                    "diversify: max tips per match",
                )
            )
            continue
        if sport_count.get(tip.sport, 0) >= policy.max_per_sport:
            rejected.append(
                RejectedTip(
                    tip.sport,
                    tip.match,
                    tip.pick,
                    tip.odds,
                    "diversify: max tips per sport",
                )
            )
            continue
        if league_count.get(tip.league_key, 0) >= policy.max_per_league:
            rejected.append(
                RejectedTip(
                    tip.sport,
                    tip.match,
                    tip.pick,
                    tip.odds,
                    "diversify: max tips per league",
                )
            )
            continue

        picked.append(tip)
        match_count[tip.match_key] = match_count.get(tip.match_key, 0) + 1
        sport_count[tip.sport] = sport_count.get(tip.sport, 0) + 1
        league_count[tip.league_key] = league_count.get(tip.league_key, 0) + 1

        if len(picked) >= policy.top_limit:
            break

    return picked


def kelly_fraction_for_tip(tip: ScoredTip, policy: TipPolicy) -> tuple[float, float]:
    """Return (raw_kelly, stake_units). Uses fractional Kelly + clamps."""
    b = tip.odds - 1.0
    p = max(0.01, min(0.99, tip.model_probability))
    q = 1.0 - p
    if b <= 0:
        return 0.0, 0.0

    raw = (b * p - q) / b
    if raw <= 0:
        return raw, 0.0

    frac = raw * policy.kelly_fraction
    stake = frac * policy.bankroll_units
    if tip.confidence < 80:
        stake = min(stake, 1.0)
    if tip.confidence < 75:
        stake = min(stake, 0.75)

    stake = max(policy.stake_min_u, min(policy.stake_max_u, stake))
    stake = round(stake * 4) / 4
    stake = max(policy.stake_min_u, min(policy.stake_max_u, stake))
    return raw, stake


def assign_stakes(selected: list[ScoredTip], policy: TipPolicy) -> float:
    total = 0.0
    for tip in selected:
        raw, stake = kelly_fraction_for_tip(tip, policy)
        tip.kelly_raw = raw
        tip.stake_u = stake
        total += stake
    return total


def build_tip_card(module_outputs: list[dict], policy: TipPolicy | None = None) -> TipCard:
    policy = policy or TipPolicy.from_env()
    rejected: list[RejectedTip] = []

    all_tips = extract_raw_pro_tips(module_outputs)
    value_tips = (
        sort_tips(filter_value_tips(all_tips))
        if policy.require_value_filter
        else all_tips
    )

    pool = value_tips if value_tips else all_tips
    scored: list[ScoredTip] = []
    for tip in pool:
        item = _normalize_scored(tip, rejected=rejected, policy=policy)
        if item is not None:
            scored.append(item)

    if policy.require_value_filter and len(all_tips) > len(value_tips):
        value_ids = {id(t) for t in value_tips}
        dropped = 0
        for tip in all_tips:
            if id(tip) in value_ids:
                continue
            dropped += 1
            if dropped <= 15:
                rejected.append(
                    RejectedTip(
                        sport=str(tip_get(tip, "sport", "?")),
                        match=str(tip_get(tip, "match") or tip_get(tip, "event") or "?"),
                        pick=str(tip_get(tip, "pick") or tip_get(tip, "selection") or "?"),
                        odds=to_float_or_none(tip_get(tip, "odds")),
                        reason="failed value filter",
                    )
                )

    scored = dedupe_best_odds(scored)
    selected = select_diversified_tips(scored, policy=policy, rejected=rejected)
    total_stake = assign_stakes(selected, policy)

    telegram = [
        t for t in selected if t.confidence >= policy.telegram_min_confidence
    ]

    publishable = len(selected) >= policy.min_publish and total_stake > 0

    log.info(
        "Tip card: raw=%s value=%s gated=%s selected=%s telegram=%s "
        "stake=%.2fu publishable=%s rejected=%s",
        len(all_tips),
        len(value_tips),
        len(scored),
        len(selected),
        len(telegram),
        total_stake,
        publishable,
        len(rejected),
    )

    return TipCard(
        all_tips=all_tips,
        value_tips=value_tips,
        gated=scored,
        selected=selected,
        rejected=rejected,
        telegram=telegram,
        publishable=publishable,
        total_stake_u=total_stake,
        policy=policy,
    )


def save_tip_card_artifacts(card: TipCard) -> dict[str, str]:
    """Persist selected card for audit / paper ledger."""
    export_dir = Path(os.getenv("EXPORT_DIR", "exports"))
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(LOCAL_TZ).strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_at": datetime.now(LOCAL_TZ).isoformat(),
        "publishable": card.publishable,
        "total_stake_u": card.total_stake_u,
        "policy": {
            "min_edge": card.policy.min_edge,
            "odds_min": card.policy.odds_min,
            "odds_max": card.policy.odds_max,
            "min_confidence": card.policy.min_confidence,
            "top_limit": card.policy.top_limit,
            "kelly_fraction": card.policy.kelly_fraction,
        },
        "selected": [
            {
                "sport": t.sport,
                "league": t.league,
                "match": t.match,
                "pick": t.pick,
                "odds": t.odds,
                "bookmaker": t.bookmaker,
                "edge": round(t.edge, 4),
                "confidence": t.confidence,
                "score": round(t.score, 3),
                "stake_u": t.stake_u,
                "kelly_raw": round(t.kelly_raw, 4),
                "model_probability": round(t.model_probability, 4),
                "market_probability": round(t.market_probability, 4),
                "signals": t.signal_names,
                "reason": t.reason,
            }
            for t in card.selected
        ],
        "rejected_sample": [
            {
                "sport": r.sport,
                "match": r.match,
                "pick": r.pick,
                "odds": r.odds,
                "reason": r.reason,
            }
            for r in card.rejected[:40]
        ],
    }

    latest_json = export_dir / "latest_tip_card.json"
    hist_json = export_dir / f"tip_card_{ts}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    latest_json.write_text(text, encoding="utf-8")
    hist_json.write_text(text, encoding="utf-8")

    ledger = export_dir / "paper_tips_ledger.csv"
    write_header = not ledger.exists()
    with ledger.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "sport",
                    "league",
                    "match",
                    "pick",
                    "odds",
                    "bookmaker",
                    "edge",
                    "confidence",
                    "stake_u",
                    "model_p",
                    "market_p",
                    "signals",
                    "status",
                ]
            )
        now = datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
        for t in card.selected:
            writer.writerow(
                [
                    now,
                    t.sport,
                    t.league,
                    t.match,
                    t.pick,
                    f"{t.odds:.3f}",
                    t.bookmaker,
                    f"{t.edge:.4f}",
                    t.confidence,
                    f"{t.stake_u:.2f}",
                    f"{t.model_probability:.4f}",
                    f"{t.market_probability:.4f}",
                    t.signal_names,
                    "OPEN",
                ]
            )

    try:
        underlying = [t.tip for t in card.selected]
        saved = save_tip_audit_log(underlying)
        if saved:
            log.info("Saved %s top pro tips to audit log", saved)
    except Exception as e:
        log.warning("save_tip_audit_log failed: %s", e)

    return {
        "latest_json": str(latest_json),
        "history_json": str(hist_json),
        "ledger": str(ledger),
    }


# ---------------------------------------------------------------------------
# Live metric helpers
# ---------------------------------------------------------------------------


def live_snapshot_count(ctx: RunContext) -> int:
    count = table_count(ctx.settings, "football_market_snapshots_v14")
    if count is not None:
        return count
    return int(os.getenv("FOOTBALL_FALLBACK_SNAPSHOTS", "0"))


def fmt_optional(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def count_csv_rows(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Post-scan step implementations
# ---------------------------------------------------------------------------


async def step_football_settlement(ctx: RunContext) -> Any:
    result = await settle_football_bets(
        ctx.settings,
        days_from=int(os.getenv("FOOTBALL_SETTLEMENT_DAYS_FROM", "3")),
    )
    log.info(
        "Football settlement finished: "
        "open=%s, sport_keys=%s, scores=%s, matched=%s, "
        "won=%s, lost=%s, void=%s, unmatched=%s, api_errors=%s",
        result.open_bets,
        result.sport_keys,
        result.score_events,
        result.matched_bets,
        result.settled_won,
        result.settled_lost,
        result.settled_void,
        result.unmatched_bets,
        result.api_errors,
    )
    return result


def step_football_result_learning(ctx: RunContext) -> Any:
    result = run_football_result_learning(ctx.settings)
    log.info(
        "Football result learning finished: "
        "discovered=%s, processed=%s, missing_score=%s, xg=%s, elo=%s, form=%s",
        result.discovered,
        result.processed,
        result.skipped_without_score,
        result.xg_updates,
        result.elo_updates,
        result.form_updates,
    )
    return result


def step_football_data_collector_v14(ctx: RunContext) -> Any:
    result = run_football_data_collector_v14(
        ctx.settings,
        closing_window_hours=float(os.getenv("FOOTBALL_CLOSING_WINDOW_HOURS", "12")),
    )
    log.info(
        "Football Data Collector v14 finished: "
        "market_added=%s, xg_added=%s, market_total=%s, xg_total=%s",
        result.market_snapshots_added,
        result.xg_rows_added,
        result.market_snapshots_total,
        result.xg_rows_total,
    )
    return result


def step_football_league_calibration(ctx: RunContext) -> Any:
    calibrated = rebuild_football_league_calibrations(ctx.settings)
    log.info("Football league calibration finished: rebuilt=%s", calibrated)
    return calibrated


def step_football_team_xg_v14(ctx: RunContext) -> Any:
    rebuilt = FootballTeamXGV14Database(ctx.settings).rebuild_all()
    log.info("Football Team xG v14 finished: rebuilt=%s", rebuilt)
    return rebuilt


def step_football_team_elo_v14(ctx: RunContext) -> Any:
    rebuilt = FootballTeamEloV14Database(ctx.settings).rebuild_from_history()
    log.info("Football Team ELO v14 finished: rebuilt=%s", rebuilt)
    return rebuilt


def step_football_feature_sync(ctx: RunContext) -> Any:
    result = run_football_learning(ctx.settings, min_samples=999999)
    log.info(
        "Football feature sync finished: synced=%s, settled=%s, open=%s",
        result.synced_features,
        result.settled_features,
        result.open_features,
    )
    return result


def step_football_meta_ai_v14(ctx: RunContext) -> Any:
    result = run_football_meta_ai_v14(ctx.settings)
    log.info(
        "Football Meta AI v14 finished: "
        "trained=%s, samples=%s, wins=%s, losses=%s, "
        "milestone=%s, model=%s, validation=%.3f",
        result.trained,
        result.samples,
        result.wins,
        result.losses,
        result.milestone,
        result.model_type or "none",
        result.validation_score,
    )
    if result.skipped_reason:
        log.info("Football Meta AI v14 skipped: %s", result.skipped_reason)
    return result


def step_football_postmatch_dataset_v14(ctx: RunContext) -> Any:
    result = rebuild_football_postmatch_dataset_v14(ctx.settings)
    log.info(
        "Football Postmatch Dataset v14 finished: "
        "discovered=%s, inserted=%s, updated=%s, missing_closing=%s, total=%s",
        result.discovered,
        result.inserted,
        result.updated,
        result.missing_closing_line,
        result.total_rows,
    )
    return result


def step_football_dataset_v15(ctx: RunContext) -> Any:
    result = rebuild_football_dataset_v15(ctx.settings)
    log.info(
        "Football Dataset v15 finished: "
        "discovered=%s, inserted=%s, updated=%s, "
        "with_closing=%s, with_xg=%s, with_elo=%s, with_form=%s, "
        "training_ready=%s, total=%s",
        result.discovered,
        result.inserted,
        result.updated,
        result.with_closing,
        result.with_xg,
        result.with_elo,
        result.with_form,
        result.training_ready,
        result.total_rows,
    )
    return result


def step_football_evaluation_v15(ctx: RunContext) -> Any:
    result = run_football_evaluation_dashboard_v15(ctx.settings)
    log.info(
        "Football Evaluation Dashboard v15 finished: "
        "total=%s, training_ready=%s, wins=%s, losses=%s, "
        "hit_rate=%s, brier=%s, log_loss=%s, avg_clv=%s, avg_consensus_safety=%s",
        result.total_rows,
        result.training_ready_rows,
        result.wins,
        result.losses,
        fmt_optional(result.hit_rate, 3),
        fmt_optional(result.brier_score, 4),
        fmt_optional(result.log_loss, 4),
        fmt_optional(result.average_clv_probability, 4),
        fmt_optional(result.average_consensus_safety, 3),
    )
    return result


def step_football_feature_importance_v15(ctx: RunContext) -> Any:
    result = run_feature_importance_v15(str(db_path(ctx.settings)))
    ranking = result.get("feature_ranking", []) if isinstance(result, dict) else []
    top_feature = ranking[0].get("feature", "n/a") if ranking else "n/a"
    log.info(
        "Football Feature Importance v15 finished: samples=%s, top_feature=%s, warning=%s",
        result.get("training_samples", 0) if isinstance(result, dict) else 0,
        top_feature,
        result.get("warning", "none") if isinstance(result, dict) else "none",
    )
    return result


def step_football_insights_v15(ctx: RunContext) -> Any:
    fi = ctx.feature_importance()
    result = run_football_insights_v15(
        dataset_report={"training_samples": ctx.dataset_training_ready()},
        feature_report=fi,
        explainability_rows=0,
    )
    log.info(
        "Football Insights v15.1 finished: samples=%s, status=%s, weight_tuning=%s",
        result.get("training_samples", 0),
        result.get("model_status", "unknown"),
        result.get("automatic_weight_tuning", False),
    )
    return result


def step_football_ai_health_v15(ctx: RunContext) -> Any:
    fi = ctx.feature_importance()
    ranking = fi.get("feature_ranking", []) or []
    top_feature = ranking[0].get("feature") if ranking else "n/a"
    result = run_ai_health_report_v15(
        dataset_samples=ctx.dataset_training_ready(),
        meta_samples=0,
        explainability_rows=0,
        top_feature=top_feature,
        missing_features=fi.get("missing_features", []) or [],
    )
    log.info(
        "Football AI Health v15.2 finished: maturity=%s, samples=%s, meta_ai=%s, weight_tuning=%s",
        result.get("model_maturity"),
        result.get("training_samples"),
        result.get("meta_ai", {}).get("status"),
        result.get("weight_tuning", {}).get("enabled"),
    )
    return result


def step_football_learning_progress_v15(ctx: RunContext) -> Any:
    result = run_learning_progress_v15(
        settled_samples=ctx.dataset_training_ready(),
        elo_available=True,
        form_available=True,
        xg_available=False,
        closing_odds_available=ctx.dataset_with_closing() > 0,
        market_snapshots_available=live_snapshot_count(ctx) > 0,
    )
    log.info(
        "Football Learning Progress v15.3 finished: "
        "samples=%s, meta_ai=%s%%, optimizer=%s%%, readiness=%s",
        result.get("settled_samples", 0),
        result.get("meta_ai", {}).get("progress_percent", 0),
        result.get("weight_optimizer", {}).get("progress_percent", 0),
        result.get("learning_readiness", "unknown"),
    )
    return result


def step_football_learning_progress_v15_4(ctx: RunContext) -> Any:
    result = run_learning_progress_v15_4(
        settled_samples=ctx.dataset_training_ready(),
        elo_available=True,
        form_available=True,
        market_available=True,
        xg_available=False,
        closing_odds_available=ctx.dataset_with_closing() > 0,
    )
    log.info(
        "Football Learning Progress v15.4 finished: samples=%s, quality=%s, readiness=%s",
        result.get("settled_samples", 0),
        result.get("data_quality_score", 0),
        result.get("learning_readiness", "unknown"),
    )
    return result


def step_football_data_quality_v15_5(ctx: RunContext) -> Any:
    result = run_data_quality_booster_v15_5(
        elo_available=True,
        form_available=True,
        market_available=True,
        xg_available=False,
        closing_odds_available=ctx.dataset_with_closing() > 0,
        settled_samples=ctx.dataset_training_ready(),
    )
    log.info(
        "Football Data Quality Booster v15.5 finished: quality=%s/100, status=%s, meta_ai_ready=%s",
        result.get("quality_score", 0),
        result.get("status", "unknown"),
        result.get("meta_ai_training_ready", False),
    )
    return result


def step_football_closing_odds_v15_6(ctx: RunContext) -> Any:
    result = run_closing_odds_collector_v15_6(
        total_samples=ctx.dataset_training_ready(),
        closing_odds_samples=ctx.dataset_with_closing(),
        market_snapshots=live_snapshot_count(ctx),
    )
    log.info(
        "Football Closing Odds Collector v15.6 finished: coverage=%s%%, status=%s",
        result.get("coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_xg_v15_7(ctx: RunContext) -> Any:
    xg_rows = table_count(ctx.settings, "football_xg_history_v14") or 0
    result = run_xg_collector_v15_7(
        total_samples=ctx.dataset_training_ready(),
        xg_samples=0,
        xg_history_rows=xg_rows,
    )
    log.info(
        "Football xG Collector v15.7 finished: coverage=%s%%, status=%s",
        result.get("coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_data_readiness_v15_8(ctx: RunContext) -> Any:
    result = run_data_readiness_v15_8(
        samples=ctx.dataset_training_ready(),
        elo=True,
        form=True,
        market=True,
        closing_odds=ctx.dataset_with_closing() > 0,
        xg=False,
    )
    log.info(
        "Football Data Readiness v15.8 finished: quality=%s/100, meta_ai_ready=%s",
        result.get("quality_score", 0),
        result.get("meta_ai_ready", False),
    )
    return result


def step_football_data_capture_v15_9(ctx: RunContext) -> Any:
    result = run_data_capture_v15_9_1(
        samples=ctx.dataset_training_ready(),
        elo_available=True,
        form_available=True,
        market_available=True,
        closing_odds_available=ctx.dataset_with_closing() > 0,
        xg_available=False,
    )
    log.info(
        "Football Data Capture Engine v15.9.1 finished: "
        "quality=%s/100, status=%s, meta_ai_ready=%s",
        result.get("quality_score", 0),
        result.get("status", "unknown"),
        result.get("meta_ai_ready", False),
    )
    return result


def step_football_closing_capture_v15_10(ctx: RunContext) -> Any:
    ready = ctx.dataset_training_ready()
    result = run_closing_odds_capture_v15_10(
        samples=ready,
        opening_odds_samples=ready,
        closing_odds_samples=ctx.dataset_with_closing(),
        market_snapshots=live_snapshot_count(ctx),
    )
    log.info(
        "Football Closing Odds Capture v15.10 finished: coverage=%s%%, status=%s",
        result.get("closing_coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_closing_line_resolver_v15_11(ctx: RunContext) -> Any:
    ready = ctx.dataset_training_ready()
    result = run_closing_line_resolver_v15_11(
        samples=ready,
        opening_odds=ready,
        market_snapshots=live_snapshot_count(ctx),
        closing_odds_found=ctx.dataset_with_closing(),
    )
    log.info(
        "Football Closing Line Resolver v15.11 finished: coverage=%s%%, status=%s",
        result.get("closing_coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_closing_storage_clv_v15_12(ctx: RunContext) -> Any:
    result = run_closing_storage_clv_v15_12(
        samples=ctx.dataset_training_ready(),
        market_snapshots=live_snapshot_count(ctx),
        closing_written=ctx.dataset_with_closing(),
        avg_clv=0.0,
    )
    log.info(
        "Football Closing Odds Storage V15.12 finished: coverage=%s%%, status=%s",
        result.get("closing_coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_closing_backfill_v15_13(ctx: RunContext) -> Any:
    result = run_closing_backfill_v15_13(
        postmatch_samples=ctx.dataset_training_ready(),
        market_snapshots=live_snapshot_count(ctx),
        closing_recovered=0,
    )
    log.info(
        "Football Closing Odds Backfill v15.13 finished: coverage=%s%%, status=%s",
        result.get("coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_snapshot_matcher_v15_14(ctx: RunContext) -> Any:
    result = run_snapshot_matcher_v15_14(
        matches=ctx.dataset_training_ready(),
        snapshots=live_snapshot_count(ctx),
        closing_matched=0,
    )
    log.info(
        "Football Closing Snapshot Matcher v15.14 finished: coverage=%s%%, status=%s",
        result.get("coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_real_closing_extractor_v15_15(ctx: RunContext) -> Any:
    result = run_real_closing_snapshot_extractor_v15_15(
        matches=ctx.dataset_training_ready(),
        snapshots=live_snapshot_count(ctx),
        closing_extracted=0,
        clv_ready=0,
    )
    log.info(
        "Football Real Closing Snapshot Extractor v15.15 finished: coverage=%s%%, status=%s",
        result.get("coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_match_id_resolver_v15_16(ctx: RunContext) -> Any:
    result = run_match_id_resolver_v15_16(
        postmatch_matches=ctx.dataset_training_ready(),
        market_snapshots=live_snapshot_count(ctx),
        matches_resolved=0,
        closing_recovered=0,
    )
    log.info(
        "Football Match ID Resolver v15.16 finished: resolved=%s%%, closing=%s%%, status=%s",
        result.get("match_resolution_percent", 0),
        result.get("closing_coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_universal_match_key_v15_17(ctx: RunContext) -> Any:
    result = run_universal_match_key_builder_v15_17(
        matches=ctx.dataset_training_ready(),
        snapshots=live_snapshot_count(ctx),
        keys_created=0,
        joins_completed=0,
    )
    log.info(
        "Football Universal Match Key Builder v15.17 finished: keys=%s%%, joins=%s%%, status=%s",
        result.get("key_coverage_percent", 0),
        result.get("join_coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_universal_join_executor_v15_18(ctx: RunContext) -> Any:
    result = run_universal_join_executor_v15_18(
        matches=ctx.dataset_training_ready(),
        snapshots=live_snapshot_count(ctx),
        keys_created=0,
        joins_completed=0,
        closing_written=0,
    )
    log.info(
        "Football Universal Join Executor v15.18 finished: joins=%s%%, closing=%s%%, status=%s",
        result.get("join_coverage_percent", 0),
        result.get("closing_coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_snapshot_schema_extractor_v15_20(ctx: RunContext) -> Any:
    result = run_snapshot_schema_extractor_v15_20(
        snapshots=live_snapshot_count(ctx),
        parsed_snapshots=0,
        fingerprints_created=0,
        join_ready=0,
    )
    log.info(
        "Football Snapshot Schema Extractor v15.20 finished: "
        "parsed=%s%%, join_ready=%s%%, status=%s",
        result.get("parse_coverage_percent", 0),
        result.get("join_ready_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_closing_odds_writer_v15_21(ctx: RunContext) -> Any:
    result = run_closing_odds_writer_v15_21(
        matches=ctx.dataset_training_ready(),
        snapshots=live_snapshot_count(ctx),
        closing_written=0,
        clv_ready=0,
    )
    log.info(
        "Football Closing Odds Writer v15.21 finished: "
        "closing_written=%s%%, clv_ready=%s%%, status=%s",
        result.get("closing_coverage_percent", 0),
        result.get("clv_coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_closing_database_join_v15_19(ctx: RunContext) -> Any:
    result = run_closing_odds_database_join_engine_v15_19(
        matches=ctx.dataset_training_ready(),
        snapshots=live_snapshot_count(ctx),
        joins_executed=0,
        closing_written=0,
        clv_calculated=0,
    )
    log.info(
        "Football Closing Odds Database Join Engine v15.19 finished: "
        "joins=%s%%, closing=%s%%, status=%s",
        result.get("join_coverage_percent", 0),
        result.get("closing_coverage_percent", 0),
        result.get("status", "unknown"),
    )
    return result


def step_football_maintenance_v14(ctx: RunContext) -> Any:
    result = run_football_maintenance_v14(
        ctx.settings,
        snapshot_retention_days=int(os.getenv("FOOTBALL_SNAPSHOT_RETENTION_DAYS", "45")),
        diagnostics_retention_days=int(
            os.getenv("FOOTBALL_DIAGNOSTICS_RETENTION_DAYS", "14")
        ),
    )
    log.info(
        "Football Maintenance v14 finished: "
        "deleted_market=%s, deleted_diagnostics=%s, settled=%s, "
        "valid_probabilities=%s, open=%s, hit_rate=%s, threshold_accuracy=%s, "
        "brier=%s, log_loss=%s, avg_clv=%s",
        result.deleted_market_snapshots,
        0,
        result.settled_samples,
        result.valid_probability_samples,
        result.open_samples,
        fmt_optional(result.hit_rate, 3),
        fmt_optional(result.threshold_accuracy, 3),
        fmt_optional(result.brier_score, 4),
        fmt_optional(result.log_loss, 4),
        fmt_optional(result.average_clv, 4),
    )
    return result


def step_multisport_learning_v2(ctx: RunContext) -> Any:
    manager = MultisportLearningV2Manager(db_path(ctx.settings))
    result = manager.run_all(export_dir=os.getenv("EXPORT_DIR", "exports"))
    log.info(
        "Multisport Learning V2 finished: sports=%s ready=%s status=%s report=%s",
        result.get("sports_completed", 0),
        result.get("sports_ready", 0),
        result.get("status", "UNKNOWN"),
        result.get("artifacts", {}).get("json", "n/a"),
    )
    return result


def step_v16_profile_pipeline(ctx: RunContext) -> Any:
    result = run_v16_profile_pipeline()
    log.info(
        "V16.16 League Profile Pipeline: profiles=%s status=%s",
        result.get("profiles", 0),
        result.get("status", "UNKNOWN"),
    )
    return result


def step_v16_integrated_cycle(ctx: RunContext) -> Any:
    v16_inputs = build_production_inputs(
        database=db_path(ctx.settings),
        module_outputs=ctx.module_outputs,
    )
    ctx.set("v16_inputs", v16_inputs)

    v16_result = run_v16_main_hook(
        ctx.module_outputs,
        cycle_id=int(os.getenv("V16_CYCLE_ID", "1")),
        previous_result=v16_inputs.previous_result,
        previous_profit=v16_inputs.previous_profit,
    )

    final_loop = v16_result.get("stages", {}).get("v16_16_loop", {})
    log.info(
        "V16 integrated final state: status=%s loop_state=%s "
        "loop_score=%s stages=%s errors=%s feedback=%s profit=%s",
        v16_result.get("status", "UNKNOWN"),
        final_loop.get("loop_state", "UNKNOWN"),
        final_loop.get("loop_score", "n/a"),
        v16_result.get("stages_completed", 0),
        len(v16_result.get("errors", [])),
        v16_inputs.previous_result or "PENDING",
        v16_inputs.previous_profit,
    )

    if env_enabled("V16_PRODUCTION_MONITORING_ENABLED", "1"):
        try:
            history_id = save_cycle(
                db_path(ctx.settings),
                v16_result,
                v16_inputs.as_dict(),
            )
            dashboard = build_dashboard(
                db_path(ctx.settings),
                export_dir=os.getenv("EXPORT_DIR", "exports"),
            )
            alerts = evaluate_alerts(
                v16_result,
                v16_inputs.as_dict(),
                export_dir=os.getenv("EXPORT_DIR", "exports"),
            )
            ctx.set("v16_dashboard_result", dashboard)
            ctx.set("v16_alert_result", alerts)
            log.info(
                "V16 production monitoring: history_id=%s dashboard=%s alerts=%s",
                history_id,
                dashboard.get("status"),
                alerts.get("count", 0),
            )
        except Exception:
            log.exception("V16 production monitoring failed")
            ctx.record_error(
                "v16_production_monitoring",
                sys.exc_info()[1] or Exception("unknown"),
            )

    return v16_result


def build_post_scan_pipeline() -> list[PipelineStep]:
    return [
        PipelineStep(
            name="football_settlement",
            env_flag="FOOTBALL_SETTLEMENT_ENABLED",
            run=step_football_settlement,
            critical=True,
            store_as="football_settlement",
        ),
        PipelineStep(
            name="football_result_learning",
            env_flag="FOOTBALL_RESULT_LEARNING_ENABLED",
            run=step_football_result_learning,
            store_as="football_result_learning",
        ),
        PipelineStep(
            name="football_data_collector_v14",
            env_flag="FOOTBALL_DATA_COLLECTOR_V14_ENABLED",
            run=step_football_data_collector_v14,
            store_as="football_data",
        ),
        PipelineStep(
            name="football_league_calibration",
            env_flag="FOOTBALL_LEAGUE_CALIBRATION_ENABLED",
            run=step_football_league_calibration,
            store_as="calibrated_leagues",
        ),
        PipelineStep(
            name="football_team_xg_v14",
            env_flag="FOOTBALL_TEAM_XG_V14_ENABLED",
            run=step_football_team_xg_v14,
            store_as="rebuilt_team_xg",
        ),
        PipelineStep(
            name="football_team_elo_v14",
            env_flag="FOOTBALL_TEAM_ELO_V14_ENABLED",
            run=step_football_team_elo_v14,
            store_as="rebuilt_team_elo",
        ),
        PipelineStep(
            name="football_feature_sync",
            env_flag="FOOTBALL_LEARNING_ENABLED",
            run=step_football_feature_sync,
            store_as="football_learning",
        ),
        PipelineStep(
            name="football_meta_ai_v14",
            env_flag="FOOTBALL_LEARNING_ENABLED",
            run=step_football_meta_ai_v14,
            store_as="football_meta_v14",
        ),
        PipelineStep(
            name="football_postmatch_dataset_v14",
            env_flag="FOOTBALL_POSTMATCH_DATASET_V14_ENABLED",
            run=step_football_postmatch_dataset_v14,
            store_as="postmatch_dataset",
        ),
        PipelineStep(
            name="football_dataset_v15",
            env_flag="FOOTBALL_DATASET_V15_ENABLED",
            run=step_football_dataset_v15,
            store_as="football_dataset_v15",
        ),
        PipelineStep(
            name="football_evaluation_v15",
            env_flag="FOOTBALL_EVALUATION_V15_ENABLED",
            run=step_football_evaluation_v15,
            store_as="football_evaluation",
        ),
        PipelineStep(
            name="football_feature_importance_v15",
            env_flag="FOOTBALL_FEATURE_IMPORTANCE_V15_ENABLED",
            run=step_football_feature_importance_v15,
            store_as="football_feature_importance",
        ),
        PipelineStep(
            name="football_insights_v15",
            env_flag="FOOTBALL_INSIGHTS_V15_ENABLED",
            run=step_football_insights_v15,
            store_as="football_insights",
        ),
        PipelineStep(
            name="football_ai_health_v15",
            env_flag="FOOTBALL_AI_HEALTH_V15_ENABLED",
            run=step_football_ai_health_v15,
            store_as="football_ai_health",
        ),
        PipelineStep(
            name="football_learning_progress_v15",
            env_flag="FOOTBALL_LEARNING_PROGRESS_V15_ENABLED",
            run=step_football_learning_progress_v15,
            store_as="learning_progress",
        ),
        PipelineStep(
            name="football_learning_progress_v15_4",
            env_flag="FOOTBALL_LEARNING_PROGRESS_V15_4_ENABLED",
            run=step_football_learning_progress_v15_4,
            store_as="learning_progress_v154",
        ),
        PipelineStep(
            name="football_data_quality_v15_5",
            env_flag="FOOTBALL_DATA_QUALITY_V15_5_ENABLED",
            run=step_football_data_quality_v15_5,
            store_as="data_quality_report",
        ),
        PipelineStep(
            name="football_closing_odds_v15_6",
            env_flag="FOOTBALL_CLOSING_ODDS_V15_6_ENABLED",
            run=step_football_closing_odds_v15_6,
            store_as="closing_odds_report",
        ),
        PipelineStep(
            name="football_xg_v15_7",
            env_flag="FOOTBALL_XG_V15_7_ENABLED",
            run=step_football_xg_v15_7,
            store_as="xg_report",
        ),
        PipelineStep(
            name="football_data_readiness_v15_8",
            env_flag="FOOTBALL_DATA_READINESS_V15_8_ENABLED",
            run=step_football_data_readiness_v15_8,
            store_as="readiness_report",
        ),
        PipelineStep(
            name="football_data_capture_v15_9",
            env_flag="FOOTBALL_DATA_CAPTURE_V15_9_ENABLED",
            run=step_football_data_capture_v15_9,
            store_as="capture_report",
        ),
        PipelineStep(
            name="football_closing_capture_v15_10",
            env_flag="FOOTBALL_CLOSING_CAPTURE_V15_10_ENABLED",
            run=step_football_closing_capture_v15_10,
            store_as="closing_capture",
        ),
        PipelineStep(
            name="football_closing_line_resolver_v15_11",
            env_flag="FOOTBALL_CLOSING_LINE_RESOLVER_V15_11_ENABLED",
            run=step_football_closing_line_resolver_v15_11,
            store_as="closing_line_report",
        ),
        PipelineStep(
            name="football_closing_storage_clv_v15_12",
            env_flag="FOOTBALL_CLOSING_STORAGE_CLV_V15_12_ENABLED",
            run=step_football_closing_storage_clv_v15_12,
            store_as="closing_storage_report",
        ),
        PipelineStep(
            name="football_closing_backfill_v15_13",
            env_flag="FOOTBALL_CLOSING_BACKFILL_V15_13_ENABLED",
            run=step_football_closing_backfill_v15_13,
            store_as="backfill_report",
        ),
        PipelineStep(
            name="football_snapshot_matcher_v15_14",
            env_flag="FOOTBALL_CLOSING_SNAPSHOT_MATCHER_V15_14_ENABLED",
            run=step_football_snapshot_matcher_v15_14,
            store_as="snapshot_match_report",
        ),
        PipelineStep(
            name="football_real_closing_extractor_v15_15",
            env_flag="FOOTBALL_REAL_CLOSING_EXTRACTOR_V15_15_ENABLED",
            run=step_football_real_closing_extractor_v15_15,
            store_as="closing_extractor_report",
        ),
        PipelineStep(
            name="football_match_id_resolver_v15_16",
            env_flag="FOOTBALL_MATCH_ID_RESOLVER_V15_16_ENABLED",
            run=step_football_match_id_resolver_v15_16,
            store_as="match_id_report",
        ),
        PipelineStep(
            name="football_universal_match_key_v15_17",
            env_flag="FOOTBALL_UNIVERSAL_MATCH_KEY_V15_17_ENABLED",
            run=step_football_universal_match_key_v15_17,
            store_as="match_key_report",
        ),
        PipelineStep(
            name="football_universal_join_executor_v15_18",
            env_flag="FOOTBALL_UNIVERSAL_JOIN_EXECUTOR_V15_18_ENABLED",
            run=step_football_universal_join_executor_v15_18,
            store_as="join_report",
        ),
        PipelineStep(
            name="football_snapshot_schema_extractor_v15_20",
            env_flag="FOOTBALL_SNAPSHOT_SCHEMA_EXTRACTOR_V15_20_ENABLED",
            run=step_football_snapshot_schema_extractor_v15_20,
            store_as="snapshot_schema_report",
        ),
        PipelineStep(
            name="football_closing_odds_writer_v15_21",
            env_flag="FOOTBALL_CLOSING_ODDS_WRITER_V15_21_ENABLED",
            run=step_football_closing_odds_writer_v15_21,
            store_as="closing_writer_report",
        ),
        PipelineStep(
            name="football_closing_database_join_v15_19",
            env_flag="FOOTBALL_CLOSING_DATABASE_JOIN_V15_19_ENABLED",
            run=step_football_closing_database_join_v15_19,
            store_as="closing_db_report",
        ),
        PipelineStep(
            name="football_maintenance_v14",
            env_flag="FOOTBALL_MAINTENANCE_V14_ENABLED",
            run=step_football_maintenance_v14,
            store_as="football_maintenance",
        ),
        PipelineStep(
            name="multisport_learning_v2",
            env_flag="MULTISPORT_LEARNING_V2_ENABLED",
            run=step_multisport_learning_v2,
            only_full_scan=False,
            store_as="multisport_v2_result",
        ),
        PipelineStep(
            name="v16_profile_pipeline",
            env_flag="V16_LEAGUE_PROFILE_ENABLED",
            run=step_v16_profile_pipeline,
            only_full_scan=False,
            store_as="v16_profile_result",
        ),
        PipelineStep(
            name="v16_integrated_cycle",
            env_flag="V16_INTEGRATED_CYCLE_ENABLED",
            run=step_v16_integrated_cycle,
            only_full_scan=False,
            store_as="v16_result",
        ),
    ]


# ---------------------------------------------------------------------------
# Reporting — tip-first
# ---------------------------------------------------------------------------


def format_tip_card_section(card: TipCard) -> str:
    lines: list[str] = []
    p = card.policy

    lines.append("\n" + "=" * 60)
    lines.append("=== TODAY'S BETTING CARD ===")
    lines.append("=" * 60)
    lines.append(
        f"Generated: {datetime.now(LOCAL_TZ).strftime('%d.%m.%Y %H:%M %Z')}"
    )
    lines.append(
        f"Policy: edge≥{p.min_edge:.0%} | odds {p.odds_min:.2f}-{p.odds_max:.2f} | "
        f"conf≥{p.min_confidence} | kelly={p.kelly_fraction:.0%}"
    )
    lines.append(
        f"Pipeline: raw={len(card.all_tips)} → value={len(card.value_tips)} → "
        f"gated={len(card.gated)} → selected={len(card.selected)}"
    )

    if not card.selected:
        lines.append("")
        lines.append("NO VALUE TODAY")
        lines.append(
            "No tip passed hard gates + diversification. "
            "Not forcing weak picks."
        )
    else:
        lines.append("")
        lines.append(
            f"PLAYABLE TIPS ({len(card.selected)})  |  "
            f"Total stake: {card.total_stake_u:.2f}u"
        )
        lines.append("-" * 60)
        for i, t in enumerate(card.selected, 1):
            lines.append(f"{i}) {t.sport} | {t.league} | {t.match}")
            lines.append(
                f"   Pick: {t.pick} @ {t.odds:.2f}"
                + (f" ({t.bookmaker})" if t.bookmaker else "")
            )
            lines.append(
                f"   Edge: {t.edge:+.1%} | Conf: {t.confidence} | "
                f"Stake: {t.stake_u:.2f}u | Score: {t.score:.1f}"
            )
            lines.append(
                f"   Model p: {t.model_probability:.1%} | "
                f"Market p: {t.market_probability:.1%} | "
                f"Kelly raw: {t.kelly_raw:.2%} | Signals: {t.signal_names}"
            )
            if t.reason:
                why = t.reason if len(t.reason) <= 180 else t.reason[:177] + "..."
                lines.append(f"   Why: {why}")
            lines.append("")

    sports_used = sorted({t.sport for t in card.selected})
    lines.append("=== RISK ===")
    lines.append(
        f"Exposure today: {card.total_stake_u:.2f}u | "
        f"Sports: {len(sports_used)} ({', '.join(sports_used) or '—'}) | "
        f"Publishable: {'YES' if card.publishable else 'NO'}"
    )
    lines.append(
        f"Caps: max {p.max_per_match}/match, {p.max_per_sport}/sport, "
        f"{p.max_per_league}/league | stake {p.stake_min_u:.2f}-{p.stake_max_u:.2f}u"
    )

    lines.append("")
    lines.append(f"=== HIGH CONFIDENCE (conf ≥ {p.telegram_min_confidence}) ===")
    if card.telegram:
        for t in card.telegram:
            lines.append(
                f"- {t.sport} | {t.match} | {t.pick} @ {t.odds:.2f} | "
                f"conf {t.confidence} | {t.stake_u:.2f}u"
            )
    else:
        lines.append("None.")

    lines.append("")
    lines.append("=== REJECTED (sample) ===")
    if not card.rejected:
        lines.append("None.")
    else:
        reason_counts: dict[str, int] = {}
        for r in card.rejected:
            key = r.reason.split(":")[0].strip()
            reason_counts[key] = reason_counts.get(key, 0) + 1
        lines.append(
            "By reason: "
            + ", ".join(
                f"{k}={v}"
                for k, v in sorted(reason_counts.items(), key=lambda x: -x[1])
            )
        )
        for r in card.rejected[:12]:
            odds_s = f"{r.odds:.2f}" if r.odds is not None else "n/a"
            lines.append(
                f"- {r.sport} | {r.match} | {r.pick} @ {odds_s} → {r.reason}"
            )
        if len(card.rejected) > 12:
            lines.append(f"... +{len(card.rejected) - 12} more")

    lines.append("")
    return "\n".join(lines) + "\n"


def format_email_card(card: TipCard) -> str:
    lines = [
        format_tip_card_section(card).rstrip(),
        "",
        "Full engine diagnostics are in the saved report file.",
        "",
    ]
    return "\n".join(lines)


def build_core_report(
    results: list,
    module_outputs: list[dict],
    card: TipCard | None = None,
) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_report(results)
    base_report_text = buffer.getvalue()

    if card is None:
        card = build_tip_card(module_outputs)

    underlying_selected = [t.tip for t in card.selected]
    try:
        legacy_top = select_top_tips(card.value_tips, limit=card.policy.top_limit)
    except Exception:
        legacy_top = underlying_selected
    try:
        legacy_rejected = rejected_tips(card.all_tips, legacy_top, limit=10)
        legacy_rejected_txt = format_rejected_report(legacy_rejected)
    except Exception:
        legacy_rejected_txt = ""

    if env_enabled("LEARNING_RETRAIN_AFTER_SCAN", "1"):
        try:
            weights = retrain_from_results()
            log.info("Learning model weights updated: %s", weights)
        except Exception as e:
            log.warning("Learning model retrain failed: %s", e)

    parts: list[str] = [
        format_tip_card_section(card),
        "\n=== LEGACY PRO REPORT (selected tips) ===\n",
    ]
    try:
        parts.append(format_pro_report(underlying_selected))
    except Exception:
        parts.append("(legacy formatter unavailable)\n")
    if legacy_rejected_txt:
        parts.append(legacy_rejected_txt)

    parts.append("\n\n=== ORIGINAL MODULE REPORT ===\n")
    parts.append(base_report_text)
    parts.append("\n\n=== ENGINE SUMMARY ===\n")

    for item in module_outputs:
        status = "OK" if item["ok"] else "FAILED"
        parts.append(
            f"- {item['sport']}: {status} ({item['duration_sec']:.2f}s)\n"
        )

    failed = [item for item in module_outputs if not item["ok"]]
    if failed:
        parts.append("\n=== FAILED MODULES ===\n")
        for item in failed:
            parts.append(f"- {item['sport']}: {item['error']}\n")

    return "".join(parts)


def section_football_ai_health(ctx: RunContext) -> str:
    health = ctx.get("football_ai_health")
    if not health:
        return ""
    explainability_count = count_csv_rows(
        Path("exports/history_football_explainability_v15.csv")
    )
    return (
        "\n\n=== FOOTBALL AI HEALTH V15.2 ===\n"
        f"Model maturity: {health.get('model_maturity', 'n/a')}\n"
        f"Training samples: {health.get('training_samples', 0)}\n"
        f"Meta AI status: {health.get('meta_ai', {}).get('status', 'n/a')}\n"
        f"Weight tuning: {health.get('weight_tuning', {}).get('enabled', False)}\n"
        f"Top feature: {health.get('top_feature', 'n/a')}\n"
        f"Explainability records: {explainability_count}\n"
    )


def section_learning_progress(ctx: RunContext) -> str:
    progress = ctx.get("learning_progress")
    if not progress:
        return ""
    return (
        "\n=== FOOTBALL LEARNING PROGRESS V15.3 ===\n"
        f"Settled samples: {progress.get('settled_samples', 0)}\n"
        f"Meta AI progress: {progress.get('meta_ai', {}).get('progress_percent', 0)}%\n"
        f"Meta AI status: {progress.get('meta_ai', {}).get('status', 'n/a')}\n"
        f"Weight Optimizer progress: "
        f"{progress.get('weight_optimizer', {}).get('progress_percent', 0)}%\n"
        f"Readiness: {progress.get('learning_readiness', 'n/a')}\n"
    )


def section_multisport_v2(ctx: RunContext) -> str:
    result = ctx.get("multisport_v2_result")
    if not result:
        return ""
    lines = [
        "\n\n=== MULTISPORT LEARNING BUNDLE V2 ===\n",
        f"Sports completed: {result.get('sports_completed', 0)}\n",
        f"Sports ready: {result.get('sports_ready', 0)}\n",
        f"Status: {result.get('status', 'UNKNOWN')}\n",
    ]
    weights = result.get("adaptive_weights", {}) or {}
    for sport, metrics in (result.get("sports") or {}).items():
        lines.append(
            f"- {sport}: samples={metrics.get('settled_bets', 0)} "
            f"| winrate={metrics.get('win_rate')} "
            f"| yield={metrics.get('yield')} "
            f"| profit={metrics.get('profit')} "
            f"| maturity={metrics.get('maturity', 'UNKNOWN')} "
            f"| health={metrics.get('ai_health', 'UNKNOWN')} "
            f"| weight={weights.get(sport, 0)}\n"
        )
    return "".join(lines)


def section_v16_profile(ctx: RunContext) -> str:
    result = ctx.get("v16_profile_result")
    if not result:
        return ""
    return (
        "\n\n=== V16.16 LEAGUE PROFILE ===\n"
        f"Profiles: {result.get('profiles', 0)}\n"
        f"Status: {result.get('status', 'UNKNOWN')}\n"
    )


def section_v16_cycle(ctx: RunContext) -> str:
    v16_result = ctx.get("v16_result")
    if not v16_result:
        return ""
    final_loop = v16_result.get("stages", {}).get("v16_16_loop", {})
    parts = [
        "\n\n=== V16 INTEGRATED AUTONOMOUS CYCLE ===\n",
        f"Cycle ID: {v16_result.get('cycle_id', 'n/a')}\n",
        f"Stages completed: {v16_result.get('stages_completed', 0)}\n",
        f"Loop state: {final_loop.get('loop_state', 'UNKNOWN')}\n",
        f"Loop score: {final_loop.get('loop_score', 'n/a')}\n",
        f"Decision: {final_loop.get('decision', 'UNKNOWN')}\n",
        f"Errors: {len(v16_result.get('errors', []))}\n",
        f"Status: {v16_result.get('status', 'UNKNOWN')}\n",
    ]
    v16_inputs = ctx.get("v16_inputs")
    if v16_inputs is not None:
        parts.append(
            f"Feedback source: {v16_inputs.source}\n"
            f"Previous result: {v16_inputs.previous_result or 'PENDING'}\n"
            f"Previous profit: {v16_inputs.previous_profit:.4f}\n"
            f"Runtime health: {v16_inputs.runtime_health:.1%}\n"
            f"Average module latency: {v16_inputs.latency_ms} ms\n"
        )
    return "".join(parts)


def section_v16_dashboard(ctx: RunContext) -> str:
    dashboard = ctx.get("v16_dashboard_result")
    if not dashboard:
        return ""
    summary = dashboard.get("summary", {}) or {}
    return (
        "\n\n=== V16 PRODUCTION DASHBOARD ===\n"
        f"Stored cycles: {summary.get('cycles', 0)}\n"
        f"Ready cycles: {summary.get('ready_cycles', 0)}\n"
        f"Error cycles: {summary.get('error_cycles', 0)}\n"
        f"Average loop score: {summary.get('average_loop_score', 0)}\n"
        f"Dashboard HTML: {dashboard.get('html', 'n/a')}\n"
    )


def section_v16_alerts(ctx: RunContext) -> str:
    alerts = ctx.get("v16_alert_result")
    if not alerts:
        return ""
    parts = [
        "\n\n=== V16 ALERTS ===\n",
        f"Active alerts: {alerts.get('count', 0)}\n",
        f"Highest severity: {alerts.get('highest_severity', 'NONE')}\n",
    ]
    for alert in alerts.get("alerts", []) or []:
        parts.append(
            f"- [{alert.get('severity', 'INFO')}] {alert.get('message', '')}\n"
        )
    return "".join(parts)


def section_pipeline_errors(ctx: RunContext) -> str:
    if not ctx.errors:
        return ""
    parts = ["\n\n=== PIPELINE ERRORS ===\n"]
    for err in ctx.errors:
        tag = "CRITICAL" if err.critical else "WARN"
        parts.append(f"- [{tag}] {err.name}: {err.message}\n")
    return "".join(parts)


def assemble_full_report(ctx: RunContext, successful_results: list) -> str:
    card = ctx.tip_card or build_tip_card(ctx.module_outputs)
    ctx.tip_card = card
    core = build_core_report(successful_results, ctx.module_outputs, card=card)
    sections = [
        core,
        audit_block_summary(ctx.settings),
        bankroll_summary(),
        performance_report(ctx.settings),
        section_football_ai_health(ctx),
        section_learning_progress(ctx),
        section_multisport_v2(ctx),
        section_v16_profile(ctx),
        section_v16_cycle(ctx),
        section_v16_dashboard(ctx),
        section_v16_alerts(ctx),
        section_pipeline_errors(ctx),
    ]
    return "".join(s for s in sections if s)


def save_report(report_text: str) -> Path:
    export_dir = Path(os.getenv("EXPORT_DIR", "exports"))
    export_dir.mkdir(parents=True, exist_ok=True)

    latest_file = export_dir / "latest_multisport_report.txt"
    latest_file.write_text(report_text, encoding="utf-8")

    if env_enabled("REPORT_SAVE_HISTORY", "1"):
        timestamp = datetime.now(LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
        archive_file = export_dir / f"multisport_report_{timestamp}.txt"
        archive_file.write_text(report_text, encoding="utf-8")

    return latest_file


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def run() -> int:
    """
    Main entry. Returns process exit code:
      0 = success
      1 = critical step / all sport modules failed
    """
    args = parse_args()
    settings = Settings.from_env()
    settings.dry_run = args.dry_run
    policy = TipPolicy.from_env()

    ctx = RunContext(settings=settings, args=args)

    restore_learning_history(settings)

    if args.sport == "all":
        selected = list(SPORT_MODULES)
    else:
        selected = [m for m in SPORT_MODULES if m.name == args.sport]

    if not selected:
        log.warning("No sport modules selected.")
        return 1

    concurrency = max(1, args.concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded_run(sport: Any) -> dict:
        async with semaphore:
            return await run_sport_module(sport, settings, args)

    module_outputs = list(
        await asyncio.gather(*(guarded_run(sport) for sport in selected))
    )
    ctx.module_outputs = module_outputs

    if all(not item["ok"] for item in module_outputs):
        ctx.record_error(
            "sport_scan",
            RuntimeError("All selected sport modules failed"),
            critical=True,
        )

    # Tip selection on scan outputs (before long maintenance).
    try:
        ctx.tip_card = build_tip_card(module_outputs, policy=policy)
        artifacts = save_tip_card_artifacts(ctx.tip_card)
        log.info("Tip artifacts: %s", artifacts)
    except Exception:
        log.exception("Tip card build failed")
        ctx.record_error("tip_card", sys.exc_info()[1] or Exception("tip_card"))

    await run_pipeline(build_post_scan_pipeline(), ctx)

    save_learning_history(settings)

    successful_results = [
        item["result"]
        for item in module_outputs
        if item["ok"] and item["result"] is not None
    ]

    report_text = assemble_full_report(ctx, successful_results)
    report_file = save_report(report_text)

    if ctx.tip_card is not None:
        print(format_tip_card_section(ctx.tip_card))
        print(f"\n[full report saved: {report_file}]\n")
    else:
        print(report_text)

    log.info("Report saved to %s", report_file)

    should_send_email = ctx.is_full_scan and not args.no_email
    if should_send_email:
        card = ctx.tip_card
        if (
            card is not None
            and policy.email_only_if_publishable
            and not card.publishable
        ):
            log.info(
                "Email skipped: no publishable tips "
                "(set EMAIL_ONLY_IF_PUBLISHABLE=0 to force)."
            )
        else:
            email_body = (
                format_email_card(card) if card is not None else report_text
            )
            prefix = (
                f"{len(card.selected)} Pro Tips"
                if card and card.selected
                else "No Value Day"
            )
            send_multisport_email(email_body, subject_prefix=prefix)

    if ctx.critical_failed:
        log.error(
            "Run finished with critical failures: %s",
            ", ".join(e.name for e in ctx.errors if e.critical),
        )
        return 1

    if any(not item["ok"] for item in module_outputs):
        log.warning(
            "Run finished with non-critical sport module failures: %s",
            ", ".join(i["sport"] for i in module_outputs if not i["ok"]),
        )

    return 0


def main() -> None:
    exit_code = asyncio.run(run())
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
