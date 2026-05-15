from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core.base import BaseSportModule
from core.config import Settings
from core.models import SportResult


class FootballModule(BaseSportModule):

    name = "football"

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parent.parent
        self.script = self.root / "main_v10_profi_betting.py"

    async def _run_engine(self, args: list[str], settings: Settings) -> SportResult:

        cmd = [sys.executable, str(self.script)] + args

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.root),
        )

        stdout, stderr = await process.communicate()

        output = stdout.decode("utf-8", errors="ignore")
        err = stderr.decode("utf-8", errors="ignore")

        if err.strip():
            output += "\n\nWARNINGS / ERRORS:\n" + err

        return SportResult(
            sport=self.name,
            report=output.strip(),
            bets=[],
        )

    async def scan(self, settings: Settings) -> SportResult:
        args = ["--no-email"]

        if settings.dry_run:
            args.append("--dry-run")

        return await self._run_engine(args, settings)

    async def analytics(self, settings: Settings) -> SportResult:
        args = [
            "--analytics",
            "--analytics-days",
            str(settings.analytics_days),
            "--no-email",
        ]

        return await self._run_engine(args, settings)

    async def backtest(self, settings: Settings, days: int) -> SportResult:
        args = [
            "--backtest",
            "--backtest-days",
            str(days),
            "--no-email",
        ]

        return await self._run_engine(args, settings)
