"""
Call this function from the existing async main.py after sport modules finish.
"""

from __future__ import annotations

import logging
from typing import Any

from v16_pipeline_manager import run_v16_integrated_cycle

log = logging.getLogger("multisport-main")


def run_v16_main_hook(
    module_outputs: list[dict[str, Any]],
    *,
    cycle_id: int = 1,
    previous_result: str | None = None,
    previous_profit: float = 0.0,
) -> dict[str, Any]:
    successful = sum(1 for item in module_outputs if item.get("ok"))
    total = len(module_outputs)
    runtime_health = successful / total if total else 0.0

    result = run_v16_integrated_cycle(
        cycle_id=cycle_id,
        previous_result=previous_result,
        previous_profit=previous_profit,
        runtime_health=runtime_health,
        execution_ready=runtime_health > 0,
    )

    log.info(
        "V16 integrated cycle finished: status=%s stages=%s errors=%s",
        result.get("status"),
        result.get("stages_completed"),
        len(result.get("errors", [])),
    )
    return result
