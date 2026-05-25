from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from core.config import Settings
from core.types import SportResult
from sports.base import SportModule


class FootballModule(SportModule):
    name = "football"

    def _engine_path(self) -> Path:
        root = Path(__file__).resolve().parents[1]

        candidates = [
            root / "main_v12_syndicate_infra_betting_engine.py",
            root / "main_v11_1_quant_pro_live_news_engine.py",
            root / "main_v11_quant_pro_betting_engine.py",
            root / "main_v10_profi_betting.py",
            root / "main_v10_profi_betting_ai_backtest_engine.py",
            root / "football_engine.py",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            "Football engine not found. Put main_v10_profi_betting.py "
            "or main_v11_1_quant_pro_live_news_engine.py in the project root."
        )

    async def _run_engine(self, args: list[str], settings: Settings) -> SportResult:
        engine = self._engine_path()

        env = os.environ.copy()

        env["AKTUALNY_BANK"] = str(settings.bank)
        env["MIN_EDGE"] = str(settings.min_edge)
        env["MAX_EDGE"] = str(settings.max_edge)
        env["MAX_ODDS"] = str(settings.max_odds)
        env["MAX_STAKE_PCT"] = str(settings.max_stake_pct)
        env["KELLY_FRAC"] = str(settings.kelly_frac)
        env["DB_FILE"] = settings.db_file

        if settings.odds_api_key:
            env["ODDS_API_KEY"] = settings.odds_api_key

        # Extra football competitions from GitHub Actions / .env
        football_extra_keys = os.getenv("FOOTBALL_EXTRA_SPORT_KEYS", "").strip()
        if football_extra_keys:
            env["FOOTBALL_EXTRA_SPORT_KEYS"] = football_extra_keys

        # Optional override: allow fully custom football keys
        football_sport_keys = os.getenv("FOOTBALL_SPORT_KEYS", "").strip()
        if football_sport_keys:
            env["FOOTBALL_SPORT_KEYS"] = football_sport_keys

        cmd = [sys.executable, str(engine)] + args

        process = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        output = process.stdout.strip()
        error = process.stderr.strip()

        # OPRAVENÉ: Odfiltrovanie otravných HTTP 404 chýb z football-data.co.uk
        if error:
            error_lines = error.splitlines()
            # Ponecháme len riadky, ktoré neobsahujú HTTP 404 (napr. chýbajúce CSV pre ligy)
            filtered_lines = [line for line in error_lines if "HTTP 404" not in line]
            error = "\n".join(filtered_lines).strip()

        message_parts = []

        if output:
            message_parts.append(output)

        if error:
            message_parts.append("\nWARNINGS / ERRORS:\n" + error)

        if process.returncode != 0:
            message_parts.append(f"\nFootball engine exited with code {process.returncode}")

        return SportResult(
            sport=self.name,
            mode=" ".join(args) if args else "scan",
            bets=[],
            message="\n".join(message_parts) or "Football engine finished with no output.",
        )

    async def scan(self, settings: Settings) -> SportResult:
        args = ["--no-email"]

        if settings.dry_run:
            args.append("--dry-run")

        return await self._run_engine(args, settings)

    async def backtest(self, settings: Settings, days: int = 180) -> SportResult:
        return await self._run_engine(
            ["--backtest", "--backtest-days", str(days), "--no-email"],
            settings,
        )

    async def analytics(self, settings: Settings) -> SportResult:
        return await self._run_engine(
            ["--analytics", "--no-email"],
            settings,
        )
    async def analytics(self, settings: Settings) -> SportResult:
        return await self._run_engine(
            ["--analytics", "--no-email"],
            settings,
        )
