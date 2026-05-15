from __future__ import annotations

import argparse
import asyncio
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO
from contextlib import redirect_stdout
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from core.config import Settings
from core.registry import get_sport, get_sports
from core.reporting import print_report

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("multisport-main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multisport betting engine")

    parser.add_argument(
        "--sport",
        choices=["all"] + sorted(get_sports().keys()),
        default=os.getenv("SPORT_MODE", "football"),
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analytics", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--backtest-days", type=int, default=int(os.getenv("BACKTEST_DAYS", "180")))

    return parser.parse_args()


def send_multisport_email(body: str) -> bool:
    gmail_user = os.getenv("GMAIL_USER", "").strip()
    gmail_password = os.getenv("GMAIL_PASSWORD", "").strip()
    gmail_receiver = os.getenv("GMAIL_RECEIVER", gmail_user).strip()

    if not gmail_user or not gmail_password or not gmail_receiver:
        log.info("Email credentials missing - multisport email skipped.")
        return False

    local_tz = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Bratislava"))
    subject = f"Multisport Betting Report - {datetime.now(local_tz).strftime('%d.%m.%Y %H:%M')}"

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

    if args.sport == "all":
        selected = list(get_sports().values())
    else:
        selected = [get_sport(args.sport)]

    results = []

    for sport in selected:
        log.info("Running sport module: %s", sport.name)

        if args.analytics:
            result = await sport.analytics(settings)
        elif args.backtest:
            result = await sport.backtest(settings, days=args.backtest_days)
        else:
            result = await sport.scan(settings)

        results.append(result)

    buffer = StringIO()

    with redirect_stdout(buffer):
        print_report(results)

    report_text = buffer.getvalue()

    print(report_text)

    if not args.dry_run and not args.analytics and not args.backtest:
        send_multisport_email(report_text)


if __name__ == "__main__":
    asyncio.run(run())
