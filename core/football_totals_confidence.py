from __future__ import annotations


def totals_publication_confidence(edge: float, xg_reliability: float) -> float | None:
    """Keep unvalidated totals in audit; never turn missing xG into 1% confidence."""
    if xg_reliability < 0.30:
        return None
    return max(1.0, min(75.0, 60.0 + float(edge) * 100.0))

