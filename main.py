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
from core.football_trainer import ensure_feature_history_table
from core.football_xg import FootballXGDatabase
from core.football_elo import FootballEloDatabase
from core.football_team_form import FootballFormDatabase

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
        and os.getenv("FOOTBALL_LEARNING_ENABLED", "1") == "1"
    ):
        try:
            football_learning = run_football_learning(
                settings,
                min_samples=int(
                    os.getenv("FOOTBALL_META_MIN_SAMPLES", "80")
                ),
            )

            log.info(
                "Football learning finished: "
                "synced=%s, settled=%s, open=%s, trained=%s",
                football_learning.synced_features,
                football_learning.settled_features,
                football_learning.open_features,
                football_learning.trained,
            )

            if football_learning.training_skipped_reason:
                log.info(
                    "Football meta training skipped: %s",
                    football_learning.training_skipped_reason,
                )

        except Exception:
            log.exception("Football learning failed")

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
