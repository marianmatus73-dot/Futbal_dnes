from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


@dataclass(frozen=True)
class CandidatePrice:
    bookmaker: str
    selection: str
    odds: float


@dataclass(frozen=True)
class CandidateOptimizationStats:
    raw_rows: int
    valid_rows: int
    unique_selections: int
    removed_rows: int


def optimize_candidate_prices(
    rows: Iterable[tuple[str, str, float]],
) -> tuple[list[CandidatePrice], CandidateOptimizationStats]:
    """
    Preserve every 1X2 selection, but keep only its best bookmaker price
    before expensive feature, ensemble and meta-model calculations.
    """
    raw_rows = 0
    valid_rows = 0
    best_by_selection: dict[str, CandidatePrice] = {}

    for bookmaker, selection, odds_value in rows:
        raw_rows += 1

        bookmaker_name = str(bookmaker or "").strip()
        selection_name = str(selection or "").strip()
        odds = safe_float(odds_value, 0.0)

        if (
            not bookmaker_name
            or not selection_name
            or odds <= 1.01
        ):
            continue

        valid_rows += 1
        previous = best_by_selection.get(selection_name)

        if previous is None or odds > previous.odds:
            best_by_selection[selection_name] = CandidatePrice(
                bookmaker=bookmaker_name,
                selection=selection_name,
                odds=odds,
            )

    optimized = sorted(
        best_by_selection.values(),
        key=lambda row: row.selection.casefold(),
    )

    return (
        optimized,
        CandidateOptimizationStats(
            raw_rows=raw_rows,
            valid_rows=valid_rows,
            unique_selections=len(optimized),
            removed_rows=max(
                0,
                valid_rows - len(optimized),
            ),
        ),
    )
