from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import csv

from core.bankroll import load_bankroll, kelly_stake_amount


@dataclass
class ProTip:
    sport: str
    league: str
    match: str
    pick: str
    odds: float
    model_probability: float
    bookmaker: str = ""
    reason: str = ""

    edge: float = 0.0
    implied_probability: float = 0.0
    confidence: int = 0
    risk: str = "medium"
    stake_units: float = 0.0
    stake_amount: float = 0.0
    created_at: str = ""


def implied_probability(odds: float) -> float:
    if odds <= 1:
        return 0.0
    return 1 / odds


def calculate_edge(model_probability: float, odds: float) -> float:
    return model_probability - implied_probability(odds)


def calculate_confidence(edge: float, model_probability: float) -> int:
    score = 50
    score += edge * 400
    score += max(0, model_probability - 0.5) * 80
    return max(1, min(100, round(score)))


def calculate_risk(confidence: int, edge: float) -> str:
    if confidence >= 80 and edge >= 0.08:
        return "low"
    if confidence >= 55 and edge >= 0.04:
        return "medium"
    return "high"


def calculate_stake_units(
    model_probability: float,
    odds: float,
    max_units: float = 3.0,
) -> float:
    b = odds - 1
    p = model_probability
    q = 1 - p

    if b <= 0:
        return 0.0

    kelly = ((b * p) - q) / b

    if kelly <= 0:
        return 0.0

    stake = kelly * 0.25 * 10
    return round(max(0.25, min(max_units, stake)), 2)


def build_pro_tip(
    *,
    sport: str,
    league: str,
    match: str,
    pick: str,
    odds: float,
    model_probability: float,
    bookmaker: str = "",
    reason: str = "",
) -> ProTip:
    imp = implied_probability(odds)
    edge = model_probability - imp
    confidence = calculate_confidence(edge, model_probability)
    risk = calculate_risk(confidence, edge)
    stake = calculate_stake_units(model_probability, odds)

    bankroll = load_bankroll()

    stake_amount = kelly_stake_amount(
        bankroll=bankroll.bankroll,
        odds=odds,
        probability=model_probability,
        kelly_fraction=bankroll.kelly_fraction,
        max_stake_percent=bankroll.max_stake_percent,
    )

    return ProTip(
        sport=sport,
        league=league,
        match=match,
        pick=pick,
        odds=odds,
        model_probability=model_probability,
        bookmaker=bookmaker,
        reason=reason,
        edge=edge,
        implied_probability=imp,
        confidence=confidence,
        risk=risk,
        stake_units=stake,
        stake_amount=stake_amount,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def filter_value_tips(
    tips: list[ProTip],
    min_edge: float = 0.04,
    min_confidence: int = 65,
) -> list[ProTip]:
    return [
        tip for tip in tips
        if tip.edge >= min_edge
        and tip.confidence >= min_confidence
        and tip.stake_units > 0
    ]


def rejected_tips(
    tips: list[ProTip],
    accepted: list[ProTip],
    limit: int = 10,
) -> list[ProTip]:
    accepted_keys = {
        (tip.sport, tip.league, tip.match, tip.pick, tip.odds)
        for tip in accepted
    }

    rejected = [
        tip for tip in tips
        if (tip.sport, tip.league, tip.match, tip.pick, tip.odds)
        not in accepted_keys
    ]

    return sort_tips(rejected)[:limit]


def sort_tips(tips: list[ProTip]) -> list[ProTip]:
    return sorted(
        tips,
        key=lambda t: (t.edge, t.confidence, t.stake_units),
        reverse=True,
    )


def save_tip_audit_log(
    tips: list[ProTip],
    path: str = "exports/pro_tip_audit.csv",
) -> int:
    if not tips:
        return 0

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    exists = file_path.exists()

    with file_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(tips[0]).keys()))

        if not exists:
            writer.writeheader()

        for tip in tips:
            writer.writerow(asdict(tip))

    return len(tips)


def format_pro_report(tips: list[ProTip]) -> str:
    if not tips:
        return (
            "\n\n=== PRO TIPPER ===\n"
            "Dnes nebol nájdený žiadny dostatočne kvalitný value bet.\n"
            "NO BET je tiež profi rozhodnutie.\n"
        )

    text = "\n\n=== PRO TIPPER VALUE BETS ===\n"

    for i, tip in enumerate(tips, start=1):
        text += format_tip_block(tip, i)

    return text


def format_rejected_report(tips: list[ProTip]) -> str:
    if not tips:
        return "\n\n=== CANDIDATES REJECTED BY PRO FILTER ===\nŽiadni odmietnutí kandidáti.\n"

    text = "\n\n=== CANDIDATES REJECTED BY PRO FILTER ===\n"

    for i, tip in enumerate(tips, start=1):
        text += format_tip_block(tip, i)

    return text


def format_tip_block(tip: ProTip, index: int) -> str:
    text = f"\n#{index} {tip.sport.upper()} | {tip.league}\n"
    text += f"Match: {tip.match}\n"
    text += f"Pick: {tip.pick}\n"
    text += f"Odds: {tip.odds:.2f}\n"
    text += f"Bookmaker: {tip.bookmaker or 'N/A'}\n"
    text += f"Model probability: {tip.model_probability:.1%}\n"
    text += f"Market probability: {tip.implied_probability:.1%}\n"
    text += f"Edge: {tip.edge:.1%}\n"
    text += f"Confidence: {tip.confidence}/100\n"
    text += f"Risk: {tip.risk}\n"
    text += f"Stake: {tip.stake_units}u\n"
    text += f"Stake amount: {tip.stake_amount:.2f}\n"

    if tip.reason:
        text += f"Reason: {tip.reason}\n"

    return text
