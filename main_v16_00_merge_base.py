# V16.00 merge base
# Built from V15.21 complete main as requested.
# Next merge layers are kept from the original module structure.
#
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sqlite3
import smtplib
from contextlib import redirect_stdout
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

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
from core.football_real_closing_snapshot_extractor_v15_15 import run_real_closing_snapshot_extractor_v15_15
from core.football_match_id_resolver_v15_16 import run_match_id_resolver_v15_16
from core.football_universal_match_key_builder_v15_17 import run_universal_match_key_builder_v15_17
from core.football_universal_join_executor_v15_18 import run_universal_join_executor_v15_18
from core.football_closing_odds_database_join_engine_v15_19 import run_closing_odds_database_join_engine_v15_19
from core.football_snapshot_schema_extractor_v15_20 import run_snapshot_schema_extractor_v15_20
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
from core.football_team_xg_v14 import (
    FootballTeamXGV14Database,
)
from core.football_team_elo_v14 import (
    FootballTeamEloV14Database,
)

from sports.football import FootballModule
from sports.tennis import TennisModule
from sports.basketball import BasketballModule
from sports.hockey import HockeyModule
from sports.baseball import BaseballModule
from sports.mma import MMAModule
from sports.nfl import NFLModule

try:
    from core.sport_quant import init_sport_db
except Exception:
    init_sport_db = None


load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("multisport-main")


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
    # Existing multisport learning history.
    "sport_bets": "exports/history_sport_bets.csv",
    "sport_bookmaker_stats": "exports/history_bookmaker_stats.csv",
    "sport_elo_ratings": "exports/history_elo_ratings.csv",

    # Football v13 persistent learning data.
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


def db_path(settings: Settings) -> Path:
    return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))


def db_connect(settings: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(settings))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
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
    return table


def import_csv_to_table(settings: Settings, table: str, csv_file: str) -> int:
    table = safe_table_name(table)
    path = Path(csv_file)

    if not path.exists():
        return 0

    with db_connect(settings) as conn:
        if not table_exists(conn, table):
            return 0

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return 0

        columns = list(rows[0].keys())
        placeholders = ",".join(["?"] * len(columns))
        col_sql = ",".join(columns)

        sql = f"""
            INSERT OR IGNORE INTO {table}
            ({col_sql})
            VALUES ({placeholders})
        """

        values = [[row.get(col, "") for col in columns] for row in rows]

        before = conn.total_changes
        conn.executemany(sql, values)
        conn.commit()

        return conn.total_changes - before


def export_table_to_csv(settings: Settings, table: str, csv_file: str) -> int:
    table = safe_table_name(table)
    path = Path(csv_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with db_connect(settings) as conn:
        if not table_exists(conn, table):
            return 0

        cursor = conn.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    return len(rows)



def init_football_v13_learning_tables(settings: Settings) -> None:
    """
    Create Football v13 learning tables before CSV restore.

    GitHub Actions uses a fresh SQLite database on every run, so these
    tables must exist before their CSV histories can be imported.
    """
    ensure_feature_history_table(db_path(settings))
    FootballXGDatabase(settings).init_db()
    FootballEloDatabase(settings).init_db()
    FootballFormDatabase(settings).init_db()
    FootballLeagueCalibrationDatabase(settings).init_db()
    FootballTeamXGV14Database(settings).init_db()
    FootballTeamEloV14Database(settings).init_db()


def ensure_football_settlement_columns(settings: Settings) -> None:
    """
    Ensure sport_bets has all Football v13 settlement columns before
    persistent CSV history is imported on a fresh GitHub Actions runner.
    """
    database = db_path(settings)
    database.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database) as conn:
        table_exists = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name='sport_bets'
            """
        ).fetchone()

        if table_exists is None:
            return

        existing_columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(sport_bets)"
            ).fetchall()
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
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE sport_bets "
                    f"ADD COLUMN {column_name} {column_type}"
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
        log.warning(
            "Could not ensure Football v13 settlement columns: %s",
            e,
        )

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

    return parser.parse_args()


def send_multisport_email(body: str) -> bool:
    gmail_user = os.getenv("GMAIL_USER", "").strip()
    gmail_password = os.getenv("GMAIL_PASSWORD", "").strip()
    gmail_receiver = os.getenv("GMAIL_RECEIVER", gmail_user).strip()

    if not gmail_user or not gmail_password or not gmail_receiver:
        log.info("Email credentials missing - multisport email skipped.")
        return False

    local_tz = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Bratislava"))
    subject = (
        f"Top 5 Pro Betting Tips - "
        f"{datetime.now(local_tz).strftime('%d.%m.%Y %H:%M')}"
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
    sport,
    settings: Settings,
    args: argparse.Namespace,
) -> dict:
    started = datetime.now()

    try:
        log.info("Running sport module: %s", sport.name)

        if args.analytics:
            result = await sport.analytics(settings)

        elif args.backtest:
            result = await sport.backtest(
                settings,
                days=args.backtest_days,
            )

        else:
            result = await sport.scan(settings)

        duration = (datetime.now() - started).total_seconds()

        return {
            "sport": sport.name,
            "ok": True,
            "duration_sec": duration,
            "result": result,
            "error": None,
        }

    except Exception as e:
        duration = (datetime.now() - started).total_seconds()
        log.exception("Sport module failed: %s", sport.name)

        return {
            "sport": sport.name,
            "ok": False,
            "duration_sec": duration,
            "result": None,
            "error": str(e),
        }


def to_float_or_none(value) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_pro_tips(module_outputs: list[dict]) -> tuple[list, list]:
    raw_tips = []

    for item in module_outputs:
        result = item.get("result")

        if not result:
            continue

        candidates = []

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

                market_probability = to_float_or_none(
                    tip.get("market_probability")
                )

                if market_probability is None:
                    market_probability = 1 / odds

                consensus = build_consensus(
                    ConsensusInput(
                        sport=tip.get("sport", item["sport"]),
                        league=tip.get("league", "Unknown"),
                        match=tip.get("match") or tip.get("event") or "Unknown",
                        pick=tip.get("pick") or tip.get("selection") or "Unknown",
                        odds=odds,
                        elo_probability=to_float_or_none(
                            tip.get("elo_probability")
                        ),
                        xg_probability=to_float_or_none(
                            tip.get("xg_probability")
                        ),
                        form_probability=to_float_or_none(
                            tip.get("form_probability")
                        ),
                        market_probability=market_probability,
                        injury_penalty=to_float_or_none(
                            tip.get("injury_penalty")
                        ) or 0.0,
                        news_penalty=to_float_or_none(
                            tip.get("news_penalty")
                        ) or 0.0,
                    )
                )

                reason_parts = []

                if tip.get("reason"):
                    reason_parts.append(str(tip.get("reason")))

                if consensus.reason:
                    reason_parts.append(consensus.reason)

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

                raw_tips.append(pro_tip)

            except Exception as e:
                log.warning("Could not convert consensus tip to ProTip: %s", e)

    all_tips = sort_tips(raw_tips)
    value_tips = filter_value_tips(all_tips)

    log.info("Extracted %s raw pro tips before value filter", len(all_tips))
    log.info("Value tips after filter: %s", len(value_tips))

    return all_tips, sort_tips(value_tips)


def build_report(results: list, module_outputs: list[dict]) -> str:
    buffer = StringIO()

    with redirect_stdout(buffer):
        print_report(results)

    base_report_text = buffer.getvalue()

    all_tips, pro_tips = extract_pro_tips(module_outputs)

    top_limit = int(os.getenv("TOP_TIPS_LIMIT", "5"))
    min_telegram_conf = int(os.getenv("TELEGRAM_MIN_CONFIDENCE", "80"))

    top_tips = select_top_tips(pro_tips, limit=top_limit)
    rejected = rejected_tips(all_tips, top_tips, limit=10)
    telegram_tips = select_telegram_tips(top_tips, min_confidence=min_telegram_conf)

    saved = save_tip_audit_log(top_tips)

    if saved:
        log.info("Saved %s top pro tips to audit log", saved)

    if os.getenv("LEARNING_RETRAIN_AFTER_SCAN", "1") == "1":
        try:
            weights = retrain_from_results()
            log.info("Learning model weights updated: %s", weights)
        except Exception as e:
            log.warning("Learning model retrain failed: %s", e)

    report_text = ""
    report_text += "\n=== TOP TIPS OF THE DAY ===\n"
    report_text += format_pro_report(top_tips)
    report_text += format_rejected_report(rejected)

    report_text += "\n\n=== HIGH CONFIDENCE TIPS ===\n"

    if telegram_tips:
        report_text += (
            f"Tips with confidence >= {min_telegram_conf}: "
            f"{len(telegram_tips)}\n"
        )
    else:
        report_text += f"No tips with confidence >= {min_telegram_conf}.\n"

    report_text += "\n\n=== ORIGINAL MODULE REPORT ===\n"
    report_text += base_report_text

    report_text += "\n\n=== ENGINE SUMMARY ===\n"

    for item in module_outputs:
        status = "OK" if item["ok"] else "FAILED"
        report_text += (
            f"- {item['sport']}: {status} "
            f"({item['duration_sec']:.2f}s)\n"
        )

    failed = [item for item in module_outputs if not item["ok"]]

    if failed:
        report_text += "\n=== FAILED MODULES ===\n"

        for item in failed:
            report_text += f"- {item['sport']}: {item['error']}\n"

    return report_text


def save_report(report_text: str) -> Path:
    export_dir = Path(os.getenv("EXPORT_DIR", "exports"))
    export_dir.mkdir(parents=True, exist_ok=True)

    latest_file = export_dir / "latest_multisport_report.txt"
    latest_file.write_text(report_text, encoding="utf-8")

    if os.getenv("REPORT_SAVE_HISTORY", "1") == "1":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = export_dir / f"multisport_report_{timestamp}.txt"
        archive_file.write_text(report_text, encoding="utf-8")

    return latest_file


async def run() -> None:
    args = parse_args()

    settings = Settings.from_env()
    settings.dry_run = args.dry_run

    restore_learning_history(settings)

    if args.sport == "all":
        selected = SPORT_MODULES
    else:
        selected = [m for m in SPORT_MODULES if m.name == args.sport]

    if not selected:
        log.warning("No sport modules selected.")
        return

    concurrency = max(1, args.concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded_run(sport):
        async with semaphore:
            return await run_sport_module(sport, settings, args)

    module_outputs = await asyncio.gather(
        *(guarded_run(sport) for sport in selected)
    )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_SETTLEMENT_ENABLED", "1") == "1"
    ):
        try:
            football_settlement = await settle_football_bets(
                settings,
                days_from=int(
                    os.getenv("FOOTBALL_SETTLEMENT_DAYS_FROM", "3")
                ),
            )

            log.info(
                "Football settlement finished: "
                "open=%s, sport_keys=%s, scores=%s, matched=%s, "
                "won=%s, lost=%s, void=%s, unmatched=%s, "
                "api_errors=%s",
                football_settlement.open_bets,
                football_settlement.sport_keys,
                football_settlement.score_events,
                football_settlement.matched_bets,
                football_settlement.settled_won,
                football_settlement.settled_lost,
                football_settlement.settled_void,
                football_settlement.unmatched_bets,
                football_settlement.api_errors,
            )

        except Exception:
            log.exception("Football settlement failed")

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_RESULT_LEARNING_ENABLED", "1") == "1"
    ):
        try:
            football_result_learning = run_football_result_learning(
                settings,
            )

            log.info(
                "Football result learning finished: "
                "discovered=%s, processed=%s, missing_score=%s, "
                "xg=%s, elo=%s, form=%s",
                football_result_learning.discovered,
                football_result_learning.processed,
                football_result_learning.skipped_without_score,
                football_result_learning.xg_updates,
                football_result_learning.elo_updates,
                football_result_learning.form_updates,
            )

        except Exception:
            log.exception("Football result learning failed")

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_DATA_COLLECTOR_V14_ENABLED", "1") == "1"
    ):
        try:
            football_data = run_football_data_collector_v14(
                settings,
                closing_window_hours=float(
                    os.getenv(
                        "FOOTBALL_CLOSING_WINDOW_HOURS",
                        "12",
                    )
                ),
            )

            log.info(
                "Football Data Collector v14 finished: "
                "market_added=%s, xg_added=%s, "
                "market_total=%s, xg_total=%s",
                football_data.market_snapshots_added,
                football_data.xg_rows_added,
                football_data.market_snapshots_total,
                football_data.xg_rows_total,
            )

        except Exception:
            log.exception("Football Data Collector v14 failed")

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_LEAGUE_CALIBRATION_ENABLED", "1") == "1"
    ):
        try:
            calibrated_leagues = rebuild_football_league_calibrations(
                settings,
            )

            log.info(
                "Football league calibration finished: rebuilt=%s",
                calibrated_leagues,
            )

        except Exception:
            log.exception("Football league calibration failed")

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_TEAM_XG_V14_ENABLED", "1") == "1"
    ):
        try:
            rebuilt_team_xg = FootballTeamXGV14Database(
                settings,
            ).rebuild_all()

            log.info(
                "Football Team xG v14 finished: rebuilt=%s",
                rebuilt_team_xg,
            )

        except Exception:
            log.exception("Football Team xG v14 failed")

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_TEAM_ELO_V14_ENABLED", "1") == "1"
    ):
        try:
            rebuilt_team_elo = FootballTeamEloV14Database(
                settings,
            ).rebuild_from_history()

            log.info(
                "Football Team ELO v14 finished: rebuilt=%s",
                rebuilt_team_elo,
            )

        except Exception:
            log.exception("Football Team ELO v14 failed")

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_LEARNING_ENABLED", "1") == "1"
    ):
        try:
            # First synchronize settled results into feature history.
            football_learning = run_football_learning(
                settings,
                min_samples=999999,
            )

            log.info(
                "Football feature sync finished: "
                "synced=%s, settled=%s, open=%s",
                football_learning.synced_features,
                football_learning.settled_features,
                football_learning.open_features,
            )

        except Exception:
            log.exception("Football feature sync failed")

        try:
            football_meta_v14 = run_football_meta_ai_v14(
                settings,
            )

            log.info(
                "Football Meta AI v14 finished: "
                "trained=%s, samples=%s, wins=%s, losses=%s, "
                "milestone=%s, model=%s, validation=%.3f",
                football_meta_v14.trained,
                football_meta_v14.samples,
                football_meta_v14.wins,
                football_meta_v14.losses,
                football_meta_v14.milestone,
                football_meta_v14.model_type or "none",
                football_meta_v14.validation_score,
            )

            if football_meta_v14.skipped_reason:
                log.info(
                    "Football Meta AI v14 skipped: %s",
                    football_meta_v14.skipped_reason,
                )

        except Exception:
            log.exception("Football Meta AI v14 failed")

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_POSTMATCH_DATASET_V14_ENABLED", "1") == "1"
    ):
        try:
            postmatch_dataset = (
                rebuild_football_postmatch_dataset_v14(
                    settings,
                )
            )

            log.info(
                "Football Postmatch Dataset v14 finished: "
                "discovered=%s, inserted=%s, updated=%s, "
                "missing_closing=%s, total=%s",
                postmatch_dataset.discovered,
                postmatch_dataset.inserted,
                postmatch_dataset.updated,
                postmatch_dataset.missing_closing_line,
                postmatch_dataset.total_rows,
            )

        except Exception:
            log.exception(
                "Football Postmatch Dataset v14 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_DATASET_V15_ENABLED", "1") == "1"
    ):
        try:
            football_dataset_v15 = (
                rebuild_football_dataset_v15(
                    settings,
                )
            )

            log.info(
                "Football Dataset v15 finished: "
                "discovered=%s, inserted=%s, updated=%s, "
                "with_closing=%s, with_xg=%s, "
                "with_elo=%s, with_form=%s, "
                "training_ready=%s, total=%s",
                football_dataset_v15.discovered,
                football_dataset_v15.inserted,
                football_dataset_v15.updated,
                football_dataset_v15.with_closing,
                football_dataset_v15.with_xg,
                football_dataset_v15.with_elo,
                football_dataset_v15.with_form,
                football_dataset_v15.training_ready,
                football_dataset_v15.total_rows,
            )

        except Exception:
            log.exception(
                "Football Dataset v15 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_EVALUATION_V15_ENABLED", "1") == "1"
    ):
        try:
            football_evaluation = (
                run_football_evaluation_dashboard_v15(
                    settings,
                )
            )

            log.info(
                "Football Evaluation Dashboard v15 finished: "
                "total=%s, training_ready=%s, wins=%s, losses=%s, "
                "hit_rate=%s, brier=%s, log_loss=%s, "
                "avg_clv=%s, avg_consensus_safety=%s",
                football_evaluation.total_rows,
                football_evaluation.training_ready_rows,
                football_evaluation.wins,
                football_evaluation.losses,
                (
                    f"{football_evaluation.hit_rate:.3f}"
                    if football_evaluation.hit_rate is not None
                    else "n/a"
                ),
                (
                    f"{football_evaluation.brier_score:.4f}"
                    if football_evaluation.brier_score is not None
                    else "n/a"
                ),
                (
                    f"{football_evaluation.log_loss:.4f}"
                    if football_evaluation.log_loss is not None
                    else "n/a"
                ),
                (
                    f"{football_evaluation.average_clv_probability:.4f}"
                    if football_evaluation.average_clv_probability is not None
                    else "n/a"
                ),
                (
                    f"{football_evaluation.average_consensus_safety:.3f}"
                    if football_evaluation.average_consensus_safety is not None
                    else "n/a"
                ),
            )

        except Exception:
            log.exception(
                "Football Evaluation Dashboard v15 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_FEATURE_IMPORTANCE_V15_ENABLED", "1") == "1"
    ):
        try:
            football_feature_importance = run_feature_importance_v15(
                str(db_path(settings)),
            )

            ranking = football_feature_importance.get(
                "feature_ranking",
                [],
            )

            top_feature = (
                ranking[0].get("feature", "n/a")
                if ranking
                else "n/a"
            )

            log.info(
                "Football Feature Importance v15 finished: "
                "samples=%s, top_feature=%s, warning=%s",
                football_feature_importance.get(
                    "training_samples",
                    0,
                ),
                top_feature,
                football_feature_importance.get(
                    "warning",
                    "none",
                ),
            )

        except Exception:
            log.exception(
                "Football Feature Importance v15 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_INSIGHTS_V15_ENABLED", "1") == "1"
    ):
        try:
            football_insights = run_football_insights_v15(
                dataset_report={
                    "training_samples": football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0,
                },
                feature_report=(
                    football_feature_importance
                    if "football_feature_importance" in locals()
                    else {}
                ),
                explainability_rows=0,
            )

            log.info(
                "Football Insights v15.1 finished: "
                "samples=%s, status=%s, weight_tuning=%s",
                football_insights.get("training_samples", 0),
                football_insights.get("model_status", "unknown"),
                football_insights.get("automatic_weight_tuning", False),
            )

        except Exception:
            log.exception(
                "Football Insights v15.1 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_AI_HEALTH_V15_ENABLED", "1") == "1"
    ):
        try:
            ranking = (
                football_feature_importance.get("feature_ranking", [])
                if "football_feature_importance" in locals()
                else []
            )

            top_feature = (
                ranking[0].get("feature")
                if ranking
                else "n/a"
            )

            football_ai_health = run_ai_health_report_v15(
                dataset_samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                meta_samples=0,
                explainability_rows=0,
                top_feature=top_feature,
                missing_features=(
                    football_feature_importance.get(
                        "missing_features",
                        [],
                    )
                    if "football_feature_importance" in locals()
                    else []
                ),
            )

            log.info(
                "Football AI Health v15.2 finished: "
                "maturity=%s, samples=%s, meta_ai=%s, weight_tuning=%s",
                football_ai_health.get("model_maturity"),
                football_ai_health.get("training_samples"),
                football_ai_health.get("meta_ai", {}).get("status"),
                football_ai_health.get("weight_tuning", {}).get("enabled"),
            )

        except Exception:
            log.exception(
                "Football AI Health v15.2 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_LEARNING_PROGRESS_V15_ENABLED", "1") == "1"
    ):
        try:
            learning_progress = run_learning_progress_v15(
                settled_samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                elo_available=True,
                form_available=True,
                xg_available=False,
                closing_odds_available=False,
                market_snapshots_available=False,
            )

            log.info(
                "Football Learning Progress v15.3 finished: "
                "samples=%s, meta_ai=%s%%, optimizer=%s%%, readiness=%s",
                learning_progress.get("settled_samples", 0),
                learning_progress.get("meta_ai", {}).get(
                    "progress_percent",
                    0,
                ),
                learning_progress.get("weight_optimizer", {}).get(
                    "progress_percent",
                    0,
                ),
                learning_progress.get(
                    "learning_readiness",
                    "unknown",
                ),
            )

        except Exception:
            log.exception(
                "Football Learning Progress v15.3 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_LEARNING_PROGRESS_V15_4_ENABLED", "1") == "1"
    ):
        try:
            learning_progress_v154 = run_learning_progress_v15_4(
                settled_samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                elo_available=True,
                form_available=True,
                market_available=True,
                xg_available=False,
                closing_odds_available=False,
            )

            log.info(
                "Football Learning Progress v15.4 finished: "
                "samples=%s, quality=%s, readiness=%s",
                learning_progress_v154.get(
                    "settled_samples",
                    0,
                ),
                learning_progress_v154.get(
                    "data_quality_score",
                    0,
                ),
                learning_progress_v154.get(
                    "learning_readiness",
                    "unknown",
                ),
            )

        except Exception:
            log.exception(
                "Football Learning Progress v15.4 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_DATA_QUALITY_V15_5_ENABLED", "1") == "1"
    ):
        try:
            data_quality_report = run_data_quality_booster_v15_5(
                elo_available=True,
                form_available=True,
                market_available=True,
                xg_available=False,
                closing_odds_available=False,
                settled_samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
            )

            log.info(
                "Football Data Quality Booster v15.5 finished: "
                "quality=%s/100, status=%s, meta_ai_ready=%s",
                data_quality_report.get("quality_score", 0),
                data_quality_report.get("status", "unknown"),
                data_quality_report.get("meta_ai_training_ready", False),
            )

        except Exception:
            log.exception(
                "Football Data Quality Booster v15.5 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_CLOSING_ODDS_V15_6_ENABLED", "1") == "1"
    ):
        try:
            closing_odds_report = run_closing_odds_collector_v15_6(
                total_samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                closing_odds_samples=(
                    football_dataset_v15.with_closing
                    if hasattr(football_dataset_v15, "with_closing")
                    else 0
                )
                if "football_dataset_v15" in locals()
                else 0,
                market_snapshots=52,
            )

            log.info(
                "Football Closing Odds Collector v15.6 finished: "
                "coverage=%s%%, status=%s",
                closing_odds_report.get("coverage_percent", 0),
                closing_odds_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football Closing Odds Collector v15.6 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_XG_V15_7_ENABLED", "1") == "1"
    ):
        try:
            xg_report = run_xg_collector_v15_7(
                total_samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                xg_samples=0,
                xg_history_rows=0,
            )

            log.info(
                "Football xG Collector v15.7 finished: "
                "coverage=%s%%, status=%s",
                xg_report.get("coverage_percent", 0),
                xg_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football xG Collector v15.7 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_DATA_READINESS_V15_8_ENABLED", "1") == "1"
    ):
        try:
            readiness_report = run_data_readiness_v15_8(
                samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                elo=True,
                form=True,
                market=True,
                closing_odds=False,
                xg=False,
            )

            log.info(
                "Football Data Readiness v15.8 finished: "
                "quality=%s/100, meta_ai_ready=%s",
                readiness_report.get("quality_score", 0),
                readiness_report.get("meta_ai_ready", False),
            )

        except Exception:
            log.exception(
                "Football Data Readiness v15.8 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_DATA_CAPTURE_V15_9_ENABLED", "1") == "1"
    ):
        try:
            capture_report = run_data_capture_v15_9_1(
                samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                elo_available=True,
                form_available=True,
                market_available=True,
                closing_odds_available=False,
                xg_available=False,
            )

            log.info(
                "Football Data Capture Engine v15.9.1 finished: "
                "quality=%s/100, status=%s, meta_ai_ready=%s",
                capture_report.get("quality_score", 0),
                capture_report.get("status", "unknown"),
                capture_report.get("meta_ai_ready", False),
            )

        except Exception:
            log.exception(
                "Football Data Capture Engine v15.9 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_CLOSING_CAPTURE_V15_10_ENABLED", "1") == "1"
    ):
        try:
            closing_capture = run_closing_odds_capture_v15_10(
                samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                opening_odds_samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                closing_odds_samples=0,
                market_snapshots=55,
            )

            log.info(
                "Football Closing Odds Capture v15.10 finished: "
                "coverage=%s%%, status=%s",
                closing_capture.get(
                    "closing_coverage_percent",
                    0,
                ),
                closing_capture.get(
                    "status",
                    "unknown",
                ),
            )

        except Exception:
            log.exception(
                "Football Closing Odds Capture v15.10 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_CLOSING_LINE_RESOLVER_V15_11_ENABLED", "1") == "1"
    ):
        try:
            closing_line_report = run_closing_line_resolver_v15_11(
                samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                opening_odds=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                market_snapshots=49,
                closing_odds_found=0,
            )

            log.info(
                "Football Closing Line Resolver v15.11 finished: "
                "coverage=%s%%, status=%s",
                closing_line_report.get("closing_coverage_percent", 0),
                closing_line_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football Closing Line Resolver v15.11 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_CLOSING_STORAGE_CLV_V15_12_ENABLED", "1") == "1"
    ):
        try:
            closing_storage_report = run_closing_storage_clv_v15_12(
                samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                market_snapshots=49,
                closing_written=0,
                avg_clv=0.0,
            )

            log.info(
                "Football Closing Odds Storage V15.12 finished: "
                "coverage=%s%%, status=%s",
                closing_storage_report.get("closing_coverage_percent", 0),
                closing_storage_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football Closing Odds Storage V15.12 failed"
            )


    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_CLOSING_BACKFILL_V15_13_ENABLED", "1") == "1"
    ):
        try:
            backfill_report = run_closing_backfill_v15_13(
                postmatch_samples=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                market_snapshots=47,
                closing_recovered=0,
            )

            log.info(
                "Football Closing Odds Backfill v15.13 finished: "
                "coverage=%s%%, status=%s",
                backfill_report.get("coverage_percent", 0),
                backfill_report.get("status", "unknown"),
            )

        except Exception:
            log.exception("Football Closing Odds Backfill v15.13 failed")


    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_CLOSING_SNAPSHOT_MATCHER_V15_14_ENABLED", "1") == "1"
    ):
        try:
            snapshot_match_report = run_snapshot_matcher_v15_14(
                matches=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                snapshots=47,
                closing_matched=0,
            )

            log.info(
                "Football Closing Snapshot Matcher v15.14 finished: "
                "coverage=%s%%, status=%s",
                snapshot_match_report.get("coverage_percent", 0),
                snapshot_match_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football Closing Snapshot Matcher v15.14 failed"
            )


    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_REAL_CLOSING_EXTRACTOR_V15_15_ENABLED", "1") == "1"
    ):
        try:
            closing_extractor_report = run_real_closing_snapshot_extractor_v15_15(
                matches=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                snapshots=47,
                closing_extracted=0,
                clv_ready=0,
            )

            log.info(
                "Football Real Closing Snapshot Extractor v15.15 finished: "
                "coverage=%s%%, status=%s",
                closing_extractor_report.get("coverage_percent", 0),
                closing_extractor_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football Real Closing Snapshot Extractor v15.15 failed"
            )


    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_MATCH_ID_RESOLVER_V15_16_ENABLED", "1") == "1"
    ):
        try:
            match_id_report = run_match_id_resolver_v15_16(
                postmatch_matches=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                market_snapshots=47,
                matches_resolved=0,
                closing_recovered=0,
            )

            log.info(
                "Football Match ID Resolver v15.16 finished: "
                "resolved=%s%%, closing=%s%%, status=%s",
                match_id_report.get("match_resolution_percent", 0),
                match_id_report.get("closing_coverage_percent", 0),
                match_id_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football Match ID Resolver v15.16 failed"
            )


    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_UNIVERSAL_MATCH_KEY_V15_17_ENABLED", "1") == "1"
    ):
        try:
            match_key_report = run_universal_match_key_builder_v15_17(
                matches=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                snapshots=47,
                keys_created=0,
                joins_completed=0,
            )

            log.info(
                "Football Universal Match Key Builder v15.17 finished: "
                "keys=%s%%, joins=%s%%, status=%s",
                match_key_report.get("key_coverage_percent", 0),
                match_key_report.get("join_coverage_percent", 0),
                match_key_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football Universal Match Key Builder v15.17 failed"
            )


    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_UNIVERSAL_JOIN_EXECUTOR_V15_18_ENABLED", "1") == "1"
    ):
        try:
            join_report = run_universal_join_executor_v15_18(
                matches=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                snapshots=47,
                keys_created=0,
                joins_completed=0,
                closing_written=0,
            )

            log.info(
                "Football Universal Join Executor v15.18 finished: "
                "joins=%s%%, closing=%s%%, status=%s",
                join_report.get("join_coverage_percent", 0),
                join_report.get("closing_coverage_percent", 0),
                join_report.get("status", "unknown"),
            )

        except Exception:
            log.exception("Football Universal Join Executor v15.18 failed")


    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_SNAPSHOT_SCHEMA_EXTRACTOR_V15_20_ENABLED", "1") == "1"
    ):
        try:
            snapshot_schema_report = run_snapshot_schema_extractor_v15_20(
                snapshots=(
                    67
                ),
                parsed_snapshots=0,
                fingerprints_created=0,
                join_ready=0,
            )

            log.info(
                "Football Snapshot Schema Extractor v15.20 finished: "
                "parsed=%s%%, join_ready=%s%%, status=%s",
                snapshot_schema_report.get("parse_coverage_percent", 0),
                snapshot_schema_report.get("join_ready_percent", 0),
                snapshot_schema_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football Snapshot Schema Extractor v15.20 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_CLOSING_ODDS_WRITER_V15_21_ENABLED", "1") == "1"
    ):
        try:
            closing_writer_report = run_closing_odds_writer_v15_21(
                matches=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                snapshots=834,
                closing_written=0,
                clv_ready=0,
            )

            log.info(
                "Football Closing Odds Writer v15.21 finished: "
                "closing_written=%s%%, clv_ready=%s%%, status=%s",
                closing_writer_report.get("closing_coverage_percent", 0),
                closing_writer_report.get("clv_coverage_percent", 0),
                closing_writer_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football Closing Odds Writer v15.21 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_CLOSING_DATABASE_JOIN_V15_19_ENABLED", "1") == "1"
    ):
        try:
            closing_db_report = run_closing_odds_database_join_engine_v15_19(
                matches=(
                    football_dataset_v15.training_ready
                    if "football_dataset_v15" in locals()
                    else 0
                ),
                snapshots=639,
                joins_executed=0,
                closing_written=0,
                clv_calculated=0,
            )

            log.info(
                "Football Closing Odds Database Join Engine v15.19 finished: "
                "joins=%s%%, closing=%s%%, status=%s",
                closing_db_report.get("join_coverage_percent", 0),
                closing_db_report.get("closing_coverage_percent", 0),
                closing_db_report.get("status", "unknown"),
            )

        except Exception:
            log.exception(
                "Football Closing Odds Database Join Engine v15.19 failed"
            )

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and os.getenv("FOOTBALL_MAINTENANCE_V14_ENABLED", "1") == "1"
    ):
        try:
            football_maintenance = run_football_maintenance_v14(
                settings,
                snapshot_retention_days=int(
                    os.getenv(
                        "FOOTBALL_SNAPSHOT_RETENTION_DAYS",
                        "45",
                    )
                ),
                diagnostics_retention_days=int(
                    os.getenv(
                        "FOOTBALL_DIAGNOSTICS_RETENTION_DAYS",
                        "14",
                    )
                ),
            )

            log.info(
                "Football Maintenance v14 finished: "
                "deleted_market=%s, deleted_diagnostics=%s, "
                "settled=%s, valid_probabilities=%s, open=%s, "
                "hit_rate=%s, threshold_accuracy=%s, "
                "brier=%s, log_loss=%s, avg_clv=%s",
                football_maintenance.deleted_market_snapshots,
                0,
                football_maintenance.settled_samples,
                football_maintenance.valid_probability_samples,
                football_maintenance.open_samples,
                (
                    f"{football_maintenance.hit_rate:.3f}"
                    if football_maintenance.hit_rate is not None
                    else "n/a"
                ),
                (
                    f"{football_maintenance.threshold_accuracy:.3f}"
                    if football_maintenance.threshold_accuracy is not None
                    else "n/a"
                ),
                (
                    f"{football_maintenance.brier_score:.4f}"
                    if football_maintenance.brier_score is not None
                    else "n/a"
                ),
                (
                    f"{football_maintenance.log_loss:.4f}"
                    if football_maintenance.log_loss is not None
                    else "n/a"
                ),
                (
                    f"{football_maintenance.average_clv:.4f}"
                    if football_maintenance.average_clv is not None
                    else "n/a"
                ),
            )

        except Exception:
            log.exception("Football Maintenance v14 failed")

    save_learning_history(settings)

    successful_results = [
        item["result"]
        for item in module_outputs
        if item["ok"] and item["result"] is not None
    ]

    report_text = build_report(successful_results, module_outputs)

    report_text += audit_block_summary(settings)
    
    report_text += bankroll_summary()

    report_text += performance_report(settings)

    if "football_ai_health" in locals():
        report_text += "\n\n=== FOOTBALL AI HEALTH V15.2 ===\n"
        explainability_count = 0
        try:
            explainability_file = Path(
                "exports/history_football_explainability_v15.csv"
            )
            if explainability_file.exists():
                with explainability_file.open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    explainability_count = max(
                        sum(1 for _ in handle) - 1,
                        0,
                    )
        except Exception:
            explainability_count = 0

        report_text += (
            f"Model maturity: {football_ai_health.get('model_maturity', 'n/a')}\n"
            f"Training samples: {football_ai_health.get('training_samples', 0)}\n"
            f"Meta AI status: {football_ai_health.get('meta_ai', {}).get('status', 'n/a')}\n"
            f"Weight tuning: {football_ai_health.get('weight_tuning', {}).get('enabled', False)}\n"
            f"Top feature: {football_ai_health.get('top_feature', 'n/a')}\n"
            f"Explainability records: {explainability_count}\n"
        )

    if "learning_progress" in locals():
        report_text += "\n=== FOOTBALL LEARNING PROGRESS V15.3 ===\n"
        report_text += (
            f"Settled samples: {learning_progress.get('settled_samples', 0)}\n"
            f"Meta AI progress: {learning_progress.get('meta_ai', {}).get('progress_percent', 0)}%\n"
            f"Meta AI status: {learning_progress.get('meta_ai', {}).get('status', 'n/a')}\n"
            f"Weight Optimizer progress: {learning_progress.get('weight_optimizer', {}).get('progress_percent', 0)}%\n"
            f"Readiness: {learning_progress.get('learning_readiness', 'n/a')}\n"
        )

    report_file = save_report(report_text)

    print(report_text)

    log.info("Report saved to %s", report_file)

    should_send_email = (
        not args.dry_run
        and not args.analytics
        and not args.backtest
        and not args.no_email
    )

    if should_send_email:
        send_multisport_email(report_text)


if __name__ == "__main__":
    asyncio.run(run())
