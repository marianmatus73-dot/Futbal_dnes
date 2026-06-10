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
    "sport_bets": "exports/history_sport_bets.csv",
    "sport_bookmaker_stats": "exports/history_bookmaker_stats.csv",
    "sport_elo_ratings": "exports/history_elo_ratings.csv",
}


def db_path(settings: Settings) -> Path:
    return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))


def db_connect(settings: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(settings))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def import_csv_to_table(settings: Settings, table: str, csv_file: str) -> int:
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

        values = [
            [row.get(col, "") for col in columns]
            for row in rows
        ]

        before = conn.total_changes
        conn.executemany(sql, values)

        return conn.total_changes - before


def export_table_to_csv(settings: Settings, table: str, csv_file: str) -> int:
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


def restore_learning_history(settings: Settings) -> None:
    if init_sport_db is not None:
        try:
            init_sport_db(settings)
        except Exception as e:
            log.warning("Could not init sport learning tables: %s", e)

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
        default=os.getenv("SPORT_MODE", "football"),
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analytics", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=int(os.getenv("BACKTEST_DAYS", "180")),
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
        f"Multisport Betting Report - "
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


async def run() -> None:
    args = parse_args()

    settings = Settings.from_env()
    settings.dry_run = args.dry_run

    restore_learning_history(settings)

    if args.sport == "all":
        selected = SPORT_MODULES
    else:
        selected = [m for m in SPORT_MODULES if m.name == args.sport]

    results = []

    for sport in selected:
        log.info("Running sport module: %s", sport.name)

        try:
            if args.analytics:
                result = await sport.analytics(settings)

            elif args.backtest:
                result = await sport.backtest(
                    settings,
                    days=args.backtest_days,
                )

            else:
                result = await sport.scan(settings)

            results.append(result)

        except Exception:
            log.exception("Sport module failed: %s", sport.name)

    save_learning_history(settings)

    buffer = StringIO()

    with redirect_stdout(buffer):
        print_report(results)

    report_text = buffer.getvalue()

    print(report_text)

    if (
        not args.dry_run
        and not args.analytics
        and not args.backtest
    ):
        send_multisport_email(report_text)


if __name__ == "__main__":
    asyncio.run(run())
