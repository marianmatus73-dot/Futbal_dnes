from __future__ import annotations

import os
import csv
import requests
from pathlib import Path


AUDIT_FILE = Path("exports/pro_tip_audit.csv")


def load_latest_high_confidence_tips(min_confidence: int = 80) -> list[dict]:
    if not AUDIT_FILE.exists():
        return []

    with AUDIT_FILE.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tips = [
        row for row in rows
        if int(float(row.get("confidence", 0) or 0)) >= min_confidence
    ]

    return tips[-5:]


def format_telegram_message(tips: list[dict]) -> str:
    if not tips:
        return "Dnes žiadny tip s confidence nad 80. NO BET."

    text = "🔥 TOP PRO TIPS\\n\\n"

    for tip in tips:
        text += (
            f"{tip.get('sport', '').upper()} | {tip.get('league', '')}\\n"
            f"{tip.get('match', '')}\\n"
            f"Pick: {tip.get('pick', '')}\\n"
            f"Odds: {tip.get('odds', '')}\\n"
            f"Edge: {float(tip.get('edge', 0)):.1%}\\n"
            f"Confidence: {tip.get('confidence', '')}/100\\n"
            f"Stake: {tip.get('stake_units', '')}u\\n\\n"
        )

    return text


def send_telegram_message(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("Telegram credentials missing.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
        },
        timeout=30,
    )

    return response.ok


if __name__ == "__main__":
    tips = load_latest_high_confidence_tips()
    message = format_telegram_message(tips)
    sent = send_telegram_message(message)

    print("Telegram sent:", sent)
