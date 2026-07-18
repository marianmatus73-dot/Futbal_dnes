
"""
V16.45 EXECUTION CONTROL ENGINE

Controls final execution after decision and stake.
"""


def execute_control(action, stake, risk_status):
    if action == "PLAY" and risk_status == "SAFE" and stake > 0:
        execution = "EXECUTE"
    else:
        execution = "HOLD"

    return {
        "action": action,
        "stake": stake,
        "risk_status": risk_status,
        "execution": execution,
        "status": "READY"
    }
