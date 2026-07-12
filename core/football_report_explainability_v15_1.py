
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def load_latest_explanation(
    db_file: str | Path,
    source_hash: str,
) -> dict[str, Any] | None:
    """
    Load a saved Football Explainability V15 decision.
    """

    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT *
            FROM football_explainability_v15
            WHERE source_hash = ?
            LIMIT 1
            """,
            (source_hash,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def format_explanation_block(
    explanation: dict[str, Any] | None,
) -> str:
    if not explanation:
        return "Explanation: unavailable"

    positive = json.loads(
        explanation.get("positive_signals") or "[]"
    )
    negative = json.loads(
        explanation.get("negative_signals") or "[]"
    )

    lines = [
        "Explanation:",
    ]

    for item in positive[:5]:
        lines.append(
            f"+ {item.get('label')}: {item.get('value')}"
        )

    for item in negative[:5]:
        lines.append(
            f"- {item.get('label')}: {item.get('value')}"
        )

    lines.extend(
        [
            f"Decision: {explanation.get('verdict')}",
            f"Confidence: {explanation.get('confidence')}",
            f"Risk: {explanation.get('risk')}",
        ]
    )

    return "\n".join(lines)
