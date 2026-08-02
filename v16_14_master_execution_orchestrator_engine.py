"""
V16.14 MASTER EXECUTION ORCHESTRATOR ENGINE
"""

def orchestrate_execution(action, action_score, execution_ready, risk_safe):
    orchestration_score=round(action_score*0.5+(1 if execution_ready else 0)*0.25+(1 if risk_safe else 0)*0.25,3)
    mode="EXECUTE_NOW" if action=="EXECUTE" and orchestration_score>=0.85 else "HOLD"
    return {
        "action":action,
        "action_score":action_score,
        "execution_ready":execution_ready,
        "risk_safe":risk_safe,
        "orchestration_score":orchestration_score,
        "execution_mode":mode,
        "execution_orchestrator_active":True,
        "status":"READY"
    }
