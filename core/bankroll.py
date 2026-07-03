from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


BANKROLL_FILE = Path("exports/bankroll.json")


@dataclass
class BankrollState:
    bankroll: float = 1000.0
    max_stake_percent: float = 0.03
    kelly_fraction: float = 0.25


def load_bankroll() -> BankrollState:
    if not BANKROLL_FILE.exists():
        return BankrollState()

    data = json.loads(BANKROLL_FILE.read_text(encoding="utf-8"))

    return BankrollState(
        bankroll=float(data.get("bankroll", 1000.0)),
        max_stake_percent=float(data.get("max_stake_percent", 0.03)),
        kelly_fraction=float(data.get("kelly_fraction", 0.25)),
    )


def save_bankroll(state: BankrollState) -> None:
    BANKROLL_FILE.parent.mkdir(parents=True, exist_ok=True)
    BANKROLL_FILE.write_text(
        json.dumps(state.__dict__, indent=2),
        encoding="utf-8",
    )


def kelly_stake_amount(
    bankroll: float,
    odds: float,
    probability: float,
    kelly_fraction: float = 0.25,
    max_stake_percent: float = 0.03,
) -> float:
    b = odds - 1
    p = probability
    q = 1 - p

    if b <= 0:
        return 0.0

    kelly = ((b * p) - q) / b

    if kelly <= 0:
        return 0.0

    raw_stake = bankroll * kelly * kelly_fraction
    max_stake = bankroll * max_stake_percent

    return round(min(raw_stake, max_stake), 2)

def bankroll_summary() -> str:
    state = load_bankroll()

    return (
        "\n=== BANKROLL SUMMARY ===\n"
        f"Bankroll: {state.bankroll:.2f}\n"
        f"Kelly fraction: {state.kelly_fraction:.2f}\n"
        f"Max stake %: {state.max_stake_percent * 100:.1f}%\n"
    )


def bankroll_dict() -> dict:
    state = load_bankroll()

    return {
        "bankroll": state.bankroll,
        "kelly_fraction": state.kelly_fraction,
        "max_stake_percent": state.max_stake_percent,
    }
