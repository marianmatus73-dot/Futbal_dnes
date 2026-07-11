from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from core.bankroll import kelly_stake_amount, load_bankroll


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

    raw_edge: float | None = None
    edge: float = 0.0
    implied_probability: float = 0.0
    confidence: int = 0
    risk: str = "medium"
    stake_units: float = 0.0
    stake_amount: float = 0.0
    created_at: str = ""


def _safe_float(
    value: object,
    default: float | None = None,
) -> float | None:
    try:
        if value is None or value == "":
            return default

        return float(value)
    except (TypeError, ValueError):
        return default


def implied_probability(odds: float) -> float:
    if odds <= 1.0:
        return 0.0

    return 1.0 / odds


def calculate_edge(
    model_probability: float,
    odds: float,
) -> float:
    return model_probability - implied_probability(odds)


def calculate_confidence(
    edge: float,
    model_probability: float,
) -> int:
    """
    Záložný confidence výpočet pre staršie športové moduly,
    ktoré ešte neposielajú vlastný confidence score.
    """

    score = 50.0
    score += edge * 400.0
    score += max(0.0, model_probability - 0.5) * 80.0

    return max(1, min(100, round(score)))


def resolve_confidence(
    *,
    model_score: float | int | None,
    edge: float,
    model_probability: float,
) -> int:
    """
    Použije confidence zo športového modulu, ak ide o skóre 20–100.

    Staršie športové moduly ukladajú do score napríklad edge * 100,
    čo býva hodnota ako 6.5 alebo 12.3. Takéto hodnoty nie sú
    confidence a preto sa ignorujú.
    """

    parsed_score = _safe_float(model_score)

    if parsed_score is not None and 20.0 <= parsed_score <= 100.0:
        return max(1, min(100, round(parsed_score)))

    return calculate_confidence(
        edge=edge,
        model_probability=model_probability,
    )


def calculate_risk(
    confidence: int,
    edge: float,
) -> str:
    if confidence >= 80 and edge >= 0.08:
        return "low"

    if confidence >= 65 and edge >= 0.04:
        return "medium"

    return "high"


def calculate_stake_units(
    model_probability: float,
    odds: float,
    max_units: float = 3.0,
) -> float:
    b = odds - 1.0
    p = model_probability
    q = 1.0 - p

    if b <= 0.0:
        return 0.0

    kelly = ((b * p) - q) / b

    if kelly <= 0.0:
        return 0.0

    stake = kelly * 0.25 * 10.0

    return round(
        max(0.25, min(max_units, stake)),
        2,
    )


def rejection_reasons(
    tip: ProTip,
    min_edge: float = 0.04,
    min_confidence: int = 65,
) -> list[str]:
    reasons: list[str] = []

    if tip.edge < min_edge:
        reasons.append(
            f"consensus edge below {min_edge:.1%}"
        )

    if tip.confidence < min_confidence:
        reasons.append(
            f"confidence below {min_confidence}"
        )

    if tip.stake_units <= 0:
        reasons.append("stake <= 0")

    if tip.risk == "high":
        reasons.append("risk high")

    return reasons


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
    raw_edge: float | None = None,
    model_score: float | int | None = None,
) -> ProTip:
    odds = float(odds)
    model_probability = max(
        0.001,
        min(0.999, float(model_probability)),
    )

    imp = implied_probability(odds)
    edge = model_probability - imp

    confidence = resolve_confidence(
        model_score=model_score,
        edge=edge,
        model_probability=model_probability,
    )

    risk = calculate_risk(
        confidence=confidence,
        edge=edge,
    )

    stake = calculate_stake_units(
        model_probability=model_probability,
        odds=odds,
    )

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
        raw_edge=raw_edge,
        edge=edge,
        implied_probability=imp,
        confidence=confidence,
        risk=risk,
        stake_units=stake,
        stake_amount=stake_amount,
        created_at=datetime.now().isoformat(
            timespec="seconds"
        ),
    )


def filter_value_tips(
    tips: list[ProTip],
    min_edge: float = 0.04,
    min_confidence: int = 65,
) -> list[ProTip]:
    return [
        tip
        for tip in tips
        if tip.edge >= min_edge
        and tip.confidence >= min_confidence
        and tip.stake_units > 0
        and tip.risk != "high"
    ]


def rejected_tips(
    tips: list[ProTip],
    accepted: list[ProTip],
    limit: int = 10,
) -> list[ProTip]:
    accepted_keys = {
        (
            tip.sport,
            tip.league,
            tip.match,
            tip.pick,
            tip.odds,
        )
        for tip in accepted
    }

    rejected = [
        tip
        for tip in tips
        if (
            tip.sport,
            tip.league,
            tip.match,
            tip.pick,
            tip.odds,
        )
        not in accepted_keys
    ]

    return sort_tips(rejected)[:limit]


def sort_tips(
    tips: list[ProTip],
) -> list[ProTip]:
    return sorted(
        tips,
        key=lambda tip: (
            tip.confidence,
            tip.edge,
            tip.stake_units,
        ),
        reverse=True,
    )


def save_tip_audit_log(
    tips: list[ProTip],
    path: str = "exports/pro_tip_audit.csv",
) -> int:
    if not tips:
        return 0

    file_path = Path(path)
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = file_path.exists()

    with file_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(asdict(tips[0]).keys()),
        )

        if not exists:
            writer.writeheader()

        for tip in tips:
            writer.writerow(asdict(tip))

    return len(tips)


def format_pro_report(
    tips: list[ProTip],
) -> str:
    if not tips:
        return (
            "\n\n=== PRO TIPPER ===\n"
            "Dnes nebol nájdený žiadny dostatočne kvalitný value bet.\n"
            "NO BET je tiež profi rozhodnutie.\n"
        )

    text = "\n\n=== PRO TIPPER VALUE BETS ===\n"

    for index, tip in enumerate(tips, start=1):
        text += format_tip_block(
            tip,
            index,
            show_rejection=False,
        )

    return text


def format_rejected_report(
    tips: list[ProTip],
) -> str:
    if not tips:
        return (
            "\n\n=== CANDIDATES REJECTED BY PRO FILTER ===\n"
            "Žiadni odmietnutí kandidáti.\n"
        )

    text = "\n\n=== CANDIDATES REJECTED BY PRO FILTER ===\n"

    for index, tip in enumerate(tips, start=1):
        text += format_tip_block(
            tip,
            index,
            show_rejection=True,
        )

    return text


def format_tip_block(
    tip: ProTip,
    index: int,
    show_rejection: bool = False,
) -> str:
    text = (
        f"\n#{index} {tip.sport.upper()} | {tip.league}\n"
        f"Match: {tip.match}\n"
        f"Pick: {tip.pick}\n"
        f"Odds: {tip.odds:.2f}\n"
        f"Bookmaker: {tip.bookmaker or 'N/A'}\n"
        f"Model probability: {tip.model_probability:.1%}\n"
        f"Market probability: {tip.implied_probability:.1%}\n"
    )

    if tip.raw_edge is not None:
        text += f"Raw edge: {tip.raw_edge:.1%}\n"

    text += (
        f"Consensus edge: {tip.edge:.1%}\n"
        f"Confidence: {tip.confidence}/100\n"
        f"Risk: {tip.risk}\n"
        f"Stake: {tip.stake_units}u\n"
        f"Stake amount: {tip.stake_amount:.2f}\n"
    )

    if show_rejection:
        reasons = rejection_reasons(tip)

        if reasons:
            text += "Rejected because:\n"

            for rejection_reason in reasons:
                text += f"- {rejection_reason}\n"

    if tip.reason:
        text += f"Reason: {tip.reason}\n"

    return text
