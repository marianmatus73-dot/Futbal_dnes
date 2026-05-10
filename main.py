from __future__ import annotations

import argparse
import asyncio
import logging
import os

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
        help="Sport to run: football, tennis, basketball, hockey, or all",
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analytics", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--backtest-days", type=int, default=int(os.getenv("BACKTEST_DAYS", "180")))

    return parser.parse_args()


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

    print_report(results)


if __name__ == "__main__":
    asyncio.run(run())
