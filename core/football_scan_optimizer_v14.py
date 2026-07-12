from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


def parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()

    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class OptimizedOddsRow:
    bookmaker: str
    selection: str
    odds: float
    snapshot_type: str


@dataclass(frozen=True)
class FootballScanOptimizerStats:
    raw_offers: int
    selected_offers: int
    hours_to_start: float | None
    mode: str


def _hours_to_start(
    commence_time: str,
) -> float | None:
    start = parse_datetime(commence_time)

    if start is None:
        return None

    return (
        start - datetime.now(timezone.utc)
    ).total_seconds() / 3600.0


def optimize_h2h_snapshot_rows(
    *,
    bookmakers: list[dict[str, Any]],
    commence_time: str,
    near_window_hours: float = 72.0,
    near_books_per_selection: int = 3,
    far_books_per_selection: int = 1,
) -> tuple[
    list[OptimizedOddsRow],
    FootballScanOptimizerStats,
]:
    """
    Reduce thousands of redundant bookmaker rows while preserving:

    - the best available price for every selection;
    - several prices close to kickoff for CLV;
    - at least one representative price for distant events.

    No selection is removed merely because its price is short or long.
    """
    offers_by_selection: dict[
        str,
        list[tuple[str, float]],
    ] = {}
    raw_offers = 0

    for bookmaker in bookmakers:
        bookmaker_name = str(
            bookmaker.get("title", "")
        ).strip()

        if not bookmaker_name:
            continue

        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue

            for outcome in market.get("outcomes", []):
                selection = str(
                    outcome.get("name", "")
                ).strip()
                odds = safe_float(
                    outcome.get("price"),
                    0.0,
                )

                if not selection or odds <= 1.01:
                    continue

                raw_offers += 1
                offers_by_selection.setdefault(
                    selection,
                    [],
                ).append(
                    (bookmaker_name, odds)
                )

    hours_to_start = _hours_to_start(
        commence_time
    )

    near_event = (
        hours_to_start is not None
        and -3.0 <= hours_to_start <= near_window_hours
    )

    limit = (
        max(1, int(near_books_per_selection))
        if near_event
        else max(1, int(far_books_per_selection))
    )
    mode = "near" if near_event else "far"

    selected_rows: list[OptimizedOddsRow] = []

    for selection, offers in offers_by_selection.items():
        # Deduplicate each bookmaker and keep its best price.
        best_by_bookmaker: dict[str, float] = {}

        for bookmaker, odds in offers:
            previous = best_by_bookmaker.get(
                bookmaker,
                0.0,
            )

            if odds > previous:
                best_by_bookmaker[bookmaker] = odds

        ranked = sorted(
            best_by_bookmaker.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        if not ranked:
            continue

        chosen: list[
            tuple[str, float, str]
        ] = []

        # Always retain the market-best price.
        best_book, best_odds = ranked[0]
        chosen.append(
            (
                best_book,
                best_odds,
                "best",
            )
        )

        if limit >= 2 and len(ranked) >= 2:
            # Retain the median-priced bookmaker, which is a useful
            # approximation of consensus without storing every book.
            median_index = len(ranked) // 2
            median_book, median_odds = ranked[
                median_index
            ]

            if median_book != best_book:
                chosen.append(
                    (
                        median_book,
                        median_odds,
                        "median",
                    )
                )

        if limit >= 3 and len(ranked) >= 3:
            # Retain the shortest price as the conservative side of the
            # market range near kickoff.
            low_book, low_odds = ranked[-1]

            if all(
                low_book != item[0]
                for item in chosen
            ):
                chosen.append(
                    (
                        low_book,
                        low_odds,
                        "low",
                    )
                )

        for bookmaker, odds, snapshot_type in chosen[
            :limit
        ]:
            selected_rows.append(
                OptimizedOddsRow(
                    bookmaker=bookmaker,
                    selection=selection,
                    odds=odds,
                    snapshot_type=snapshot_type,
                )
            )

    return (
        selected_rows,
        FootballScanOptimizerStats(
            raw_offers=raw_offers,
            selected_offers=len(selected_rows),
            hours_to_start=hours_to_start,
            mode=mode,
        ),
    )
