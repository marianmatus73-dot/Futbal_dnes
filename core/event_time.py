from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def parse_datetime(value: Any) -> datetime | None:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_closing_window(
    commence_time: Any,
    *,
    captured_at: datetime | None = None,
    window_hours: float = 12.0,
) -> bool:
    commence = parse_datetime(commence_time)
    if commence is None:
        return False
    captured = captured_at or datetime.now(timezone.utc)
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    else:
        captured = captured.astimezone(timezone.utc)
    hours = (commence - captured).total_seconds() / 3600.0
    return 0.0 <= hours <= max(0.0, float(window_hours))
