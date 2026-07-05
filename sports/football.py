from __future__ import annotations

import asyncio
import csv
import os
import subprocess
import sys
from pathlib import Path

from core.config import Settings
from core.types import SportResult
from core.adaptive_weights import (
    sport_weight,
    bookmaker_weight,
    league_weight,
)
from sports.base import SportModule
from sports.football_signals import enrich_football_tip


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
            "Football engine not found. Put main_v12_syndicate_infra_betting_engine.py "
            "or another supported football engine in the project root."
        )

    def _export_dir(self) -> Path:
        return Path(os.getenv("EXPORT_DIR", "exports"))

    def _latest_csv_candidates(self) -> list[Path]:
        export_dir = self._export_dir()

        candidates = [
            export_dir / "top_bets.csv",
            export_dir / "football_top_bets.csv",
            export_dir / "value_bets.csv",
            export_dir / "football_value_bets.csv",
            export_dir / "bets.csv",
            export_dir / "football_bets.csv",
            export_dir / "pro_tip_audit.csv",
        ]

        return [path for path in candidates if path.exists()]

    def _normalize_tip_row(self, row: dict) -> dict | None:
        def pick_first(*names: str, default: str = ""):
            for name in names:
                value = row.get(name)
                if value not in (None, ""):
                    return value
            return default

        # 1. OCHRANA: Ak riadok explicitne obsahuje iný šport, preskočíme ho
        row_sport = str(row.get("sport", "")).lower()
        if row_sport and row_sport != "football":
            return None

        league = pick_first("league", "competition", "sport_key", default="Unknown")
        
        # 2. OCHRANA: Ak je v názve ligy iný šport, úplne ho odignorujeme
        other_sports = ["baseball", "tennis", "basketball", "hockey", "mma", "nfl", "americanfootball"]
        if any(x in league.lower() for x in other_sports):
            return None

        odds = pick_first(
            "odds",
            "best_odds",
            "bookmaker_odds",
            "price",
            "kurz",
            default="",
        )

        pick = pick_first(
            "pick",
            "selection",
            "bet",
            "market",
            "tip",
            default="",
        )

        match = pick_first(
            "match",
            "event",
            "fixture",
            "game",
            "home_away",
            default="",
        )

        home_team = pick_first("home_team", "home", "home_name", default="")
        away_team = pick_first("away_team", "away", "away_name", default="")

        if not match and home_team and away_team:
            match = f"{home_team} vs {away_team}"

        probability = pick_first(
            "model_probability",
            "probability",
            "win_probability",
            "prob",
            "model_prob",
            default="",
        )

        if not odds or not pick:
            return None

        tip = {
            "sport": "football",
            "league": league,
            "match": match or "Unknown",
            "pick": pick,
            "odds": odds,
            "model_probability": probability,
            "bookmaker": pick_first("bookmaker", "book", "site", default=""),
            "reason": pick_first("reason", "note", "notes", "edge_reason", default=""),
            "market_probability": pick_first("market_probability", "market_prob", "implied_probability", default=""),
            "elo_probability": pick_first("elo_probability", "elo_prob", default=""),
            "xg_probability": pick_first("xg_probability", "xg_prob", default=""),
            "form_probability": pick_first("form_probability", "form_prob", default=""),
            "injury_penalty": pick_first("injury_penalty", default="0"),
            "news_penalty": pick_first("news_penalty", default="0"),
            "home_xg": pick_first("home_xg", "xg_home", default=""),
            "away_xg": pick_first("away_xg", "xg_away", default=""),
            "form_score": pick_first("form_score", default=""),
        }

        return enrich_football_tip(tip)

    def _load_exported_tips(self) -> list[dict]:
        tips: list[dict] = []

        for csv_file in self._latest_csv_candidates():
            try:
                with csv_file.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        tip = self._normalize_tip_row(row)

                        if tip:
                            tips.append(tip)

            except Exception:
                continue

        seen = set()
        unique_tips = []

        for tip in tips:
            key = (
                tip.get("league"),
                tip.get("match"),
                tip.get("pick"),
                tip.get("odds"),
            )

            if key in seen:
                continue

            seen.add(key)
            unique_tips.append(tip)

        return unique_tips

    def _build_env(self, settings: Settings) -> dict:
        env = os.environ.copy()

        env["AKTUALNY_BANK"] = str(settings.bank)
        env["MIN_EDGE"] = str(settings.min_edge)
        env["MAX_EDGE"] = str(settings.max_edge)
        env["MAX_ODDS"] = str(settings.max_odds)
        env["MAX_STAKE_PCT"] = str(settings.max_stake_pct)
        env["KELLY_FRAC"] = str(settings.kelly_frac)
        env["DB_FILE"] = settings.db_file

        env["BANKROLL"] = os.getenv("BANKROLL", str(settings.bank))
        env["KELLY_FRACTION"] = os.getenv("KELLY_FRACTION", "0.25")
        env["MAX_STAKE_PERCENT"] = os.getenv("MAX_STAKE_PERCENT", "0.03")

        if settings.odds_api_key:
            env["ODDS_API_KEY"] = settings.odds_api_key

        passthrough_keys = [
            "FOOTBALL_EXTRA_SPORT_KEYS",
            "FOOTBALL_SPORT_KEYS",
            "SPORT_KEY_AUTO_DISCOVERY",
            "CACHE_DIR",
            "EXPORT_DIR",
            "CACHE_TTL_SECONDS",
            "HTTP_TIMEOUT",
            "LOOKAHEAD_HOURS",
            "LOCAL_TZ",
            "LOG_LEVEL",
            "MIN_PROB",
            "TOP_N_REPORT",
            "MAX_BETS_PER_DAY",
            "MAX_DAILY_EXPOSURE_PCT",
            "MAX_MATCH_EXPOSURE_PCT",
            "MAX_LEAGUE_EXPOSURE_PCT",
            "LONGSHOT_MAX_ODDS",
            "LONGSHOT_MIN_EDGE",
            "LONGSHOT_MIN_PROB",
            "MARKET_BLEND_WEIGHT",
            "MIN_MARKET_EDGE",
            "MIN_BOOKMAKERS_AGREE",
            "REQUIRE_MARKET_AGREEMENT",
            "PROB_SHRINK",
            "DIXON_COLES_RHO",
            "RECENT_MATCHES",
            "MAX_GOALS_GRID",
            "AI_FILTER_ENABLED",
            "MIN_AI_PROB",
            "MIN_AI_EDGE",
            "ELO_ENABLED",
            "ELO_BASE",
            "ELO_K",
            "ELO_HOME_ADV",
            "META_LAYER_ENABLED",
            "META_MIN_CONFIDENCE",
            "META_REDUCE_BELOW",
            "SNAPSHOT_ODDS",
            "CLV_LOOKBACK_HOURS",
            "CLV_MODEL_MIN_SAMPLES",
            "LINE_MOVE_GOOD_PCT",
            "LINE_MOVE_BAD_PCT",
            "LIVE_MODE_ENABLED",
            "LIVE_REQUIRE_STEAM",
            "LIVE_MIN_EDGE",
            "LIVE_MAX_MINUTE",
            "INJURY_NEWS_ENABLED",
            "INJURY_SOURCE_FILE",
        ]

        for key in passthrough_keys:
            value = os.getenv(key)

            if value is not None:
                env[key] = value

        return env

    def _clean_error_output(self, error: str) -> str:
        if not error:
            return ""

        ignored_fragments = [
            "HTTP 404",
            "404 Client Error",
            "No such file or directory: 'injury_news_sample.json'",
        ]

        lines = error.splitlines()
        filtered_lines = [
            line for line in lines
            if not any(fragment in line for fragment in ignored_fragments)
        ]

        return "\n".join(filtered_lines).strip()

    async def _run_engine(self, args: list[str], settings: Settings) -> SportResult:
        engine = self._engine_path()
        env = self._build_env(settings)

        before_tips_count = len(self._load_exported_tips())

        cmd = [sys.executable, str(engine)] + args

        process = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        output = process.stdout.strip()
        error = self._clean_error_output(process.stderr.strip())

        tips = self._load_exported_tips()

        message_parts = []

        if output:
            message_parts.append(output)

        if error:
            message_parts.append("\nWARNINGS / ERRORS:\n" + error)

        if process.returncode != 0:
            message_parts.append(f"\nFootball engine exited with code {process.returncode}")

        new_tips_count = max(0, len(tips) - before_tips_count)
        message_parts.append(
            f"\nFootball consensus tips loaded: {len(tips)} "
            f"(new after run approx: {new_tips_count})"
        )

        return SportResult(
            sport=self.name,
            mode=" ".join(args) if args else "scan",
            bets=tips,
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
