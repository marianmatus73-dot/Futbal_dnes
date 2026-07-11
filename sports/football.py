from __future__ import annotations

import asyncio
import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.adaptive_weights import (
    bookmaker_weight,
    league_weight,
    sport_weight,
)
from core.confidence_model import ConfidenceInput, confidence_score
from core.config import Settings
from core.monte_carlo import (
    format_monte_carlo_reason,
    monte_carlo_score,
    simulate_single_bet,
)
from core.types import SportResult
from sports.base import SportModule
from sports.football_signals import enrich_football_tip


def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    try:
        if value is None or value == "":
            return default

        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, value))


def stable_seed(*parts: Any) -> int:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


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
            "Football engine not found. Put "
            "main_v12_syndicate_infra_betting_engine.py "
            "or another supported football engine "
            "in the project root."
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

        existing = [
            path
            for path in candidates
            if path.exists()
        ]

        return sorted(
            existing,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def _resolve_confidence(
        self,
        *,
        league: str,
        match: str,
        pick: str,
        bookmaker: str,
        odds: float,
        probability: float,
        row: dict,
    ) -> tuple[int, str]:
        """
        Vytvorí jednotné confidence 1–100 pre hlavný Pro Tipper.

        Priorita vstupov:
        1. meta_confidence z futbalového enginu,
        2. Bayesian posterior,
        3. expected CLV,
        4. adaptive sport/league/bookmaker history,
        5. Monte Carlo risk score.
        """

        raw_edge = probability * odds - 1.0

        market_edge = safe_float(
            row.get("market_edge"),
            raw_edge,
        )
        market_edge = float(market_edge or 0.0)

        bayes_probability = safe_float(
            row.get("bayes_prob"),
            0.50,
        )
        bayes_probability = clamp(
            float(bayes_probability or 0.50),
            0.01,
            0.99,
        )

        expected_clv = safe_float(
            row.get("expected_clv_pct"),
            0.0,
        )
        expected_clv = float(expected_clv or 0.0)

        meta_confidence = safe_float(
            row.get("meta_confidence"),
            None,
        )

        current_sport_weight = sport_weight(self.name)
        current_bookmaker_weight = bookmaker_weight(bookmaker)
        current_league_weight = league_weight(league)

        calculated_confidence = confidence_score(
            ConfidenceInput(
                edge=raw_edge,
                consensus=clamp(
                    market_edge,
                    0.0,
                    0.20,
                ),
                bayesian=bayes_probability,
                clv=expected_clv,
                bookmaker_weight=current_bookmaker_weight,
                sport_weight=current_sport_weight,
                league_weight=current_league_weight,
                samples=0,
            )
        )

        if meta_confidence is not None:
            if 0.0 <= meta_confidence <= 1.0:
                engine_confidence = round(
                    meta_confidence * 100.0
                )
            else:
                engine_confidence = round(
                    meta_confidence
                )

            engine_confidence = max(
                1,
                min(100, engine_confidence),
            )

            base_confidence = round(
                engine_confidence * 0.60
                + calculated_confidence * 0.40
            )
        else:
            base_confidence = calculated_confidence

        mc = simulate_single_bet(
            probability=probability,
            odds=odds,
            seed=stable_seed(
                league,
                match,
                pick,
                bookmaker,
                odds,
            ),
        )

        mc_result_score = monte_carlo_score(mc)

        final_confidence = round(
            base_confidence * 0.80
            + mc_result_score * 0.20
        )

        final_confidence = max(
            1,
            min(100, final_confidence),
        )

        detail = (
            f"Football confidence={final_confidence}/100"
            f" | base={base_confidence}"
            f" | calculated={calculated_confidence}"
            f" | Bayesian={bayes_probability:.1%}"
            f" | expected CLV={expected_clv:+.2%}"
            f" | sport weight={current_sport_weight:.4f}"
            f" | league weight={current_league_weight:.4f}"
            f" | bookmaker weight={current_bookmaker_weight:.4f}"
            f" | {format_monte_carlo_reason(mc)}"
        )

        return final_confidence, detail

    def _normalize_tip_row(
        self,
        row: dict,
    ) -> dict | None:
        def pick_first(
            *names: str,
            default: Any = "",
        ) -> Any:
            for name in names:
                value = row.get(name)

                if value not in (None, ""):
                    return value

            return default

        row_sport = str(
            row.get("sport", "")
        ).lower().strip()

        if row_sport and row_sport != "football":
            return None

        league = str(
            pick_first(
                "league",
                "competition",
                "sport_key",
                default="Unknown",
            )
        )

        other_sports = [
            "baseball",
            "tennis",
            "basketball",
            "hockey",
            "mma",
            "nfl",
            "americanfootball",
        ]

        if any(
            sport_name in league.lower()
            for sport_name in other_sports
        ):
            return None

        odds_raw = pick_first(
            "odds",
            "best_odds",
            "bookmaker_odds",
            "price",
            "kurz",
            default="",
        )

        odds = safe_float(odds_raw)

        if odds is None or odds <= 1.0:
            return None

        pick = str(
            pick_first(
                "pick",
                "selection",
                "bet",
                "tip",
                default="",
            )
        )

        market = str(
            pick_first(
                "market",
                default="h2h",
            )
        )

        match = str(
            pick_first(
                "match",
                "event",
                "fixture",
                "game",
                "home_away",
                "zapas",
                default="",
            )
        )

        home_team = str(
            pick_first(
                "home_team",
                "home",
                "home_name",
                default="",
            )
        )

        away_team = str(
            pick_first(
                "away_team",
                "away",
                "away_name",
                default="",
            )
        )

        if not match and home_team and away_team:
            match = f"{home_team} vs {away_team}"

        probability_raw = pick_first(
            "model_probability",
            "probability",
            "win_probability",
            "prob",
            "model_prob",
            "prob_final",
            "ensemble_prob",
            default="",
        )

        probability = safe_float(
            probability_raw
        )

        if not pick or probability is None:
            return None

        probability = clamp(
            probability,
            0.001,
            0.999,
        )

        bookmaker = str(
            pick_first(
                "bookmaker",
                "book",
                "site",
                default="",
            )
        )

        existing_reason = str(
            pick_first(
                "reason",
                "note",
                "notes",
                "edge_reason",
                "explanation",
                default="",
            )
        )

        confidence, confidence_reason = self._resolve_confidence(
            league=league,
            match=match or "Unknown",
            pick=pick,
            bookmaker=bookmaker,
            odds=odds,
            probability=probability,
            row=row,
        )

        raw_edge = probability * odds - 1.0

        reason_parts = []

        if existing_reason:
            reason_parts.append(existing_reason)

        reason_parts.extend(
            [
                f"Football raw edge {raw_edge:.1%}",
                confidence_reason,
            ]
        )

        tip = {
            "sport": "football",
            "league": league,
            "match": match or "Unknown",
            "pick": pick,
            "market": market,
            "odds": odds,
            "model_probability": probability,
            "model_score": confidence,
            "score": confidence,
            "bookmaker": bookmaker,
            "reason": " | ".join(reason_parts),
            "raw_edge": raw_edge,
            "market_probability": pick_first(
                "market_probability",
                "market_prob",
                "implied_probability",
                "prob_market",
                default="",
            ),
            "elo_probability": pick_first(
                "elo_probability",
                "elo_prob",
                default="",
            ),
            "xg_probability": pick_first(
                "xg_probability",
                "xg_prob",
                "prob_model",
                default="",
            ),
            "form_probability": pick_first(
                "form_probability",
                "form_prob",
                default="",
            ),
            "injury_penalty": pick_first(
                "injury_penalty",
                "injury_news_risk",
                default="0",
            ),
            "news_penalty": pick_first(
                "news_penalty",
                default="0",
            ),
            "home_xg": pick_first(
                "home_xg",
                "xg_home",
                "lh",
                default="",
            ),
            "away_xg": pick_first(
                "away_xg",
                "xg_away",
                "la",
                default="",
            ),
            "form_score": pick_first(
                "form_score",
                default="",
            ),
            "bayes_probability": pick_first(
                "bayes_prob",
                default="",
            ),
            "meta_confidence": pick_first(
                "meta_confidence",
                default="",
            ),
            "expected_clv": pick_first(
                "expected_clv_pct",
                default="",
            ),
            "steam_score": pick_first(
                "steam_score",
                default="",
            ),
            "bookmaker_grade": pick_first(
                "bookmaker_grade",
                default="",
            ),
        }

        enriched = enrich_football_tip(tip)

        # Ochrana, aby enrich funkcia neprepísala nové confidence.
        enriched["model_score"] = confidence
        enriched["score"] = confidence
        enriched["raw_edge"] = raw_edge

        if confidence_reason not in str(
            enriched.get("reason", "")
        ):
            enriched_reason = str(
                enriched.get("reason", "")
            ).strip()

            enriched["reason"] = " | ".join(
                part
                for part in [
                    enriched_reason,
                    confidence_reason,
                ]
                if part
            )

        return enriched

    def _load_exported_tips(self) -> list[dict]:
        tips: list[dict] = []

        for csv_file in self._latest_csv_candidates():
            try:
                with csv_file.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as file:
                    reader = csv.DictReader(file)

                    for row in reader:
                        tip = self._normalize_tip_row(row)

                        if tip:
                            tips.append(tip)

            except Exception:
                continue

        seen: set[tuple[Any, ...]] = set()
        unique_tips: list[dict] = []

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

        unique_tips.sort(
            key=lambda tip: (
                safe_float(
                    tip.get("model_score"),
                    0.0,
                )
                or 0.0,
                safe_float(
                    tip.get("raw_edge"),
                    0.0,
                )
                or 0.0,
            ),
            reverse=True,
        )

        return unique_tips

    def _build_env(
        self,
        settings: Settings,
    ) -> dict:
        env = os.environ.copy()

        env["AKTUALNY_BANK"] = str(settings.bank)
        env["MIN_EDGE"] = str(settings.min_edge)
        env["MAX_EDGE"] = str(settings.max_edge)
        env["MAX_ODDS"] = str(settings.max_odds)
        env["MAX_STAKE_PCT"] = str(settings.max_stake_pct)
        env["KELLY_FRAC"] = str(settings.kelly_frac)
        env["DB_FILE"] = settings.db_file

        env["BANKROLL"] = os.getenv(
            "BANKROLL",
            str(settings.bank),
        )
        env["KELLY_FRACTION"] = os.getenv(
            "KELLY_FRACTION",
            "0.25",
        )
        env["MAX_STAKE_PERCENT"] = os.getenv(
            "MAX_STAKE_PERCENT",
            "0.03",
        )

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
            "MONTE_CARLO_SIMULATIONS",
        ]

        for key in passthrough_keys:
            value = os.getenv(key)

            if value is not None:
                env[key] = value

        return env

    def _clean_error_output(
        self,
        error: str,
    ) -> str:
        if not error:
            return ""

        ignored_fragments = [
            "HTTP 404",
            "404 Client Error",
            "No such file or directory: 'injury_news_sample.json'",
        ]

        lines = error.splitlines()

        filtered_lines = [
            line
            for line in lines
            if not any(
                fragment in line
                for fragment in ignored_fragments
            )
        ]

        return "\n".join(
            filtered_lines
        ).strip()

    async def _run_engine(
        self,
        args: list[str],
        settings: Settings,
    ) -> SportResult:
        engine = self._engine_path()
        env = self._build_env(settings)

        before_tips_count = len(
            self._load_exported_tips()
        )

        cmd = [
            sys.executable,
            str(engine),
            *args,
        ]

        process = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        output = process.stdout.strip()
        error = self._clean_error_output(
            process.stderr.strip()
        )

        tips = self._load_exported_tips()

        message_parts: list[str] = []

        if output:
            message_parts.append(output)

        if error:
            message_parts.append(
                "\nWARNINGS / ERRORS:\n" + error
            )

        if process.returncode != 0:
            message_parts.append(
                "\nFootball engine exited with code "
                f"{process.returncode}"
            )

        new_tips_count = max(
            0,
            len(tips) - before_tips_count,
        )

        message_parts.append(
            f"\nFootball adaptive/Monte Carlo tips loaded: "
            f"{len(tips)} "
            f"(new after run approx: {new_tips_count})"
        )

        return SportResult(
            sport=self.name,
            mode=" ".join(args) if args else "scan",
            bets=tips,
            message=(
                "\n".join(message_parts)
                or "Football engine finished with no output."
            ),
        )

    async def scan(
        self,
        settings: Settings,
    ) -> SportResult:
        args = ["--no-email"]

        if settings.dry_run:
            args.append("--dry-run")

        return await self._run_engine(
            args,
            settings,
        )

    async def backtest(
        self,
        settings: Settings,
        days: int = 180,
    ) -> SportResult:
        return await self._run_engine(
            [
                "--backtest",
                "--backtest-days",
                str(days),
                "--no-email",
            ],
            settings,
        )

    async def analytics(
        self,
        settings: Settings,
    ) -> SportResult:
        return await self._run_engine(
            [
                "--analytics",
                "--no-email",
            ],
            settings,
        )
