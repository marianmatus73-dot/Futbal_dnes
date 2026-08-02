"""
V16.15 MASTER EXECUTION MONITOR ENGINE
"""

def monitor_execution(execution_mode, orchestration_score, health_score, latency_ms):
    monitor_score = round(
        orchestration_score * 0.5 +
        health_score * 0.3 +
        (1.0 if latency_ms <= 100 else 0.8) * 0.2,
        3
    )

    system_state = "HEALTHY" if monitor_score >= 0.9 else "CHECK"

    return {
        "execution_mode": execution_mode,
        "orchestration_score": orchestration_score,
        "health_score": health_score,
        "latency_ms": latency_ms,
        "monitor_score": monitor_score,
        "system_state": system_state,
        "execution_monitor_active": True,
        "status": "READY"
    }