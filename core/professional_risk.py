from __future__ import annotations

import math
import os
import sqlite3
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

from core.config import Settings
from core.sport_policy import sport_policy
from core.sport_context import SportContextDatabase
from core.types import Bet, SportResult


@dataclass
class RiskSummary:
    candidates: int = 0
    accepted: int = 0
    rejected: int = 0
    daily_exposure: float = 0.0
    drawdown_paused: bool = False


def _settled_profile(settings: Settings, sport: str) -> tuple[int, float]:
    db = Path(settings.db_file or "bets.db")
    if not db.exists():
        return 0, .50
    with sqlite3.connect(db) as conn:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sport_bets'"
        ).fetchone() is None:
            return 0, .50
        row = conn.execute(
            """
            SELECT COUNT(*), SUM(CASE WHEN UPPER(result) IN ('WON','WIN','V') THEN 1 ELSE 0 END)
            FROM sport_bets
            WHERE sport=? AND UPPER(COALESCE(result,'')) IN ('WON','WIN','V','LOST','LOSS','P')
            """,
            (sport,),
        ).fetchone()
    samples = int(row[0] or 0)
    return samples, float(row[1] or 0) / samples if samples else .50


def calibrated_probability(probability: float, samples: int, hit_rate: float) -> float:
    probability = max(.01, min(.99, float(probability)))
    # Historical evidence is learned only from settled rows. Young sports are
    # strongly shrunk towards a neutral prior instead of receiving fake trust.
    evidence = samples / (samples + 150.0)
    baseline = (hit_rate * samples + .50 * 50.0) / (samples + 50.0)
    return max(.01, min(.99, probability * evidence + baseline * (1.0 - evidence)))


def conservative_probability(probability: float, samples: int, z: float = 1.28) -> float:
    effective_samples = max(20, samples)
    uncertainty = z * math.sqrt(probability * (1.0 - probability) / effective_samples)
    return max(.01, probability - uncertainty)


def apply_professional_risk_controls(
    outputs: list[dict], settings: Settings
) -> RiskSummary:
    summary = RiskSummary()
    peak = float(os.getenv("BANKROLL_PEAK", str(settings.bank)) or settings.bank)
    max_drawdown = float(os.getenv("MAX_BANKROLL_DRAWDOWN_PCT", "0.15"))
    summary.drawdown_paused = peak > 0 and settings.bank < peak * (1.0 - max_drawdown)
    daily_limit = settings.bank * float(os.getenv("MAX_DAILY_EXPOSURE_PCT", "0.05"))
    db = Path(settings.db_file or "bets.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS professional_risk_allocations (
                allocation_key TEXT PRIMARY KEY, allocation_date TEXT NOT NULL,
                sport TEXT NOT NULL, league TEXT, event TEXT, selection TEXT,
                stake REAL NOT NULL, created_at TEXT NOT NULL
            )
            """
        )
        today = datetime.now(timezone.utc).date().isoformat()
        accepted_daily = float(conn.execute(
            "SELECT COALESCE(SUM(stake), 0) FROM professional_risk_allocations "
            "WHERE allocation_date=?", (today,)
        ).fetchone()[0] or 0.0)
        allocated_events = {
            (str(row[0]), str(row[1]), str(row[2]))
            for row in conn.execute(
                "SELECT sport, league, event FROM professional_risk_allocations "
                "WHERE allocation_date=?", (today,)
            ).fetchall()
        }
        sport_allocations = {
            str(row[0]): float(row[1] or 0.0)
            for row in conn.execute(
                "SELECT sport, SUM(stake) FROM professional_risk_allocations "
                "WHERE allocation_date=? GROUP BY sport", (today,)
            ).fetchall()
        }
    context_db = SportContextDatabase(settings)
    for output in outputs:
        result = output.get("result")
        if not isinstance(result, SportResult):
            continue
        policy = sport_policy(result.sport)
        samples, hit_rate = _settled_profile(settings, result.sport)
        sport_limit = settings.bank * policy.max_sport_exposure_pct
        sport_exposure = sport_allocations.get(result.sport, 0.0)
        accepted: list[Bet] = []
        seen_events: set[tuple[str, str]] = set()

        for bet in sorted(result.bets, key=lambda item: (item.score, item.edge), reverse=True):
            summary.candidates += 1
            calibrated = calibrated_probability(bet.prob_final, samples, hit_rate)
            context = context_db.latest(
                result.sport, bet.event, bet.external_event_id
            )
            if context.verified:
                calibrated -= context.injury_impact + context.suspension_impact
                if context.travel_km is not None:
                    calibrated -= min(.02, max(0.0, context.travel_km) / 500000.0)
                if result.sport == "baseball" and context.starting_pitcher_confirmed:
                    calibrated += context.starting_pitcher_edge
                calibrated = max(.01, min(.99, calibrated))
            lower = conservative_probability(calibrated, samples)
            conservative_edge = lower * bet.odds - 1.0
            event_key = (bet.league, bet.event)
            daily_event_key = (result.sport, bet.league, bet.event)
            confidence = int(round(bet.score))
            stake_cap = settings.bank * policy.max_stake_pct
            stake = min(float(bet.stake), stake_cap)

            allowed = (
                not summary.drawdown_paused
                and policy.min_odds <= bet.odds <= policy.max_odds
                and policy.min_edge <= conservative_edge <= policy.max_edge
                and confidence >= policy.min_confidence
                and event_key not in seen_events
                and daily_event_key not in allocated_events
                and len(accepted) < policy.max_tips
                and accepted_daily + stake <= daily_limit
                and sport_exposure + stake <= sport_limit
            )
            if not allowed:
                summary.rejected += 1
                continue

            bet.prob_final = calibrated
            bet.edge = conservative_edge
            bet.stake = round(stake, 2)
            accepted.append(bet)
            seen_events.add(event_key)
            accepted_daily += stake
            sport_exposure += stake
            sport_allocations[result.sport] = sport_exposure
            summary.accepted += 1
            allocation_key = hashlib.sha256(
                "|".join((result.sport, bet.league, bet.event, bet.selection, bet.start_time)).encode()
            ).hexdigest()[:32]
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO professional_risk_allocations (
                        allocation_key, allocation_date, sport, league, event,
                        selection, stake, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        allocation_key, today, result.sport, bet.league, bet.event,
                        bet.selection, bet.stake,
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    ),
                )
            allocated_events.add(daily_event_key)

        result.bets = accepted

    summary.daily_exposure = round(accepted_daily, 2)
    return summary

