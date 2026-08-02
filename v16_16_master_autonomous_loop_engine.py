"""
V16.16 MASTER AUTONOMOUS LOOP ENGINE
"""

def autonomous_loop(system_ready, monitor_score, decision, cycle_id):
    loop_score = round(
        (1.0 if system_ready else 0.0) * 0.4 +
        monitor_score * 0.4 +
        (1.0 if decision == "EXECUTE_NOW" else 0.5) * 0.2,
        3
    )

    loop_state = "AUTONOMOUS_ACTIVE" if loop_score >= 0.9 else "STANDBY"

    return {
        "cycle_id": cycle_id,
        "system_ready": system_ready,
        "monitor_score": monitor_score,
        "decision": decision,
        "loop_score": loop_score,
        "loop_state": loop_state,
        "autonomous_loop_active": True,
        "status": "READY"
    }
