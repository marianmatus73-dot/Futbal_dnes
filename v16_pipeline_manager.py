"""
Integrated V16.00–V16.16 pipeline manager.

The manager passes real outputs between engines. Previous settled feedback is
optional; when it is unavailable, feedback-dependent stages are marked PENDING
instead of inventing a WIN or profit.
"""

from __future__ import annotations

from typing import Any

from v16_system_state import V16SystemState
from v16_00_master_integration import run_master_cycle
from v16_01_master_validation_layer import validate_master
from v16_02_master_feedback_fusion_layer import feedback_fusion
from v16_03_master_learning_update_engine import learning_update
from v16_04_master_strategy_adaptation_engine import strategy_adaptation
from v16_05_master_decision_policy_update_engine import policy_update
from v16_06_master_autonomous_decision_sync_engine import decision_sync
from v16_07_master_execution_alignment_engine import execution_alignment
from v16_08_master_action_control_engine import action_control
from v16_09_master_result_analysis_engine import result_analysis
from v16_10_master_intelligence_recalibration_engine import intelligence_recalibration
from v16_11_master_predictive_intelligence_alignment_engine import predictive_alignment
from v16_12_master_scenario_forecast_engine import scenario_forecast
from v16_13_master_optimal_action_engine import select_optimal_action
from v16_14_master_execution_orchestrator_engine import orchestrate_execution
from v16_15_master_execution_monitor_engine import monitor_execution
from v16_16_master_autonomous_loop_engine import autonomous_loop


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_v16_integrated_cycle(
    *,
    cycle_id: int = 1,
    previous_result: str | None = None,
    previous_profit: float = 0.0,
    runtime_health: float = 1.0,
    latency_ms: int = 50,
    execution_ready: bool = True,
) -> dict[str, Any]:
    state = V16SystemState(cycle_id=cycle_id)

    try:
        master = run_master_cycle()
        state.record("v16_00_master", master)

        intelligence = master["intelligence"]
        execution = master["execution"]
        risk = master["risk"]
        master_system = master["master_system"]

        intelligence_score = _mean([
            min(float(intelligence["prediction"]), 1.0),
            min(float(intelligence["reasoning"]), 1.0),
            min(float(intelligence["decision_score"]), 1.0),
        ])

        validation = validate_master(
            intelligence_score=intelligence_score,
            decision_score=min(float(intelligence["decision_score"]), 1.0),
            execution_ready=execution_ready,
            risk_safe=risk["status"] == "SAFE",
            stability=master_system["stability"],
        )
        state.record("v16_01_validation", validation)

        # Feedback is based only on a supplied settled previous result.
        if previous_result in {"WIN", "LOSS"}:
            feedback = feedback_fusion(
                validation_score=validation["ai_confidence_index"],
                execution_result=previous_result,
                profit=float(previous_profit),
            )
            state.record("v16_02_feedback", feedback)

            learning_signal = float(feedback["learning_signal"])
            update = learning_update(
                learning_signal=learning_signal,
                old_model_weight=1.25,
                old_strategy_weight=1.20,
            )
            state.record("v16_03_learning", update)
            model_weight = float(update["new_model_weight"])
            strategy_weight = float(update["new_strategy_weight"])
        else:
            state.record("v16_02_feedback", {
                "execution_result": "PENDING",
                "profit": 0.0,
                "model_update_ready": False,
                "status": "PENDING",
            })
            state.record("v16_03_learning", {
                "weights_updated": False,
                "status": "SKIPPED_NO_SETTLED_RESULT",
            })
            learning_signal = 0.0
            model_weight = 1.25
            strategy_weight = 1.20

        strategy = strategy_adaptation(
            model_weight=model_weight,
            strategy_weight=strategy_weight,
            learning_signal=learning_signal,
        )
        state.record("v16_04_strategy", strategy)

        policy = policy_update(
            strategy_score=float(strategy["strategy_score"]),
            strategy_mode=strategy["strategy_mode"],
            decision_threshold=min(float(intelligence["decision_score"]), 1.0),
        )
        state.record("v16_05_policy", policy)

        sync = decision_sync(
            policy_score=float(policy["policy_score"]),
            confidence=validation["ai_confidence_index"] / 100,
            risk_safe=risk["status"] == "SAFE",
            execution_ready=execution_ready,
        )
        state.record("v16_06_sync", sync)

        alignment = execution_alignment(
            final_action=sync["final_action"],
            execution_score=float(execution["execution_score"]),
            risk_control=1.0 if risk["status"] == "SAFE" else 0.0,
            timing_ready=1.0 if execution_ready else 0.0,
        )
        state.record("v16_07_alignment", alignment)

        action = action_control(
            execution_mode=alignment["execution_mode"],
            action=sync["final_action"],
            expected_score=float(alignment["alignment_score"]),
        )
        # In integrated live mode this is a control decision, not a real settled result.
        action["outcome"] = "PENDING"
        state.record("v16_08_action", action)

        analysis = result_analysis(
            outcome=previous_result if previous_result in {"WIN", "LOSS"} else "PENDING",
            execution_score=float(alignment["alignment_score"]),
            profit=float(previous_profit) if previous_result in {"WIN", "LOSS"} else 0.0,
        )
        state.record("v16_09_analysis", analysis)

        recalibration = intelligence_recalibration(
            performance_score=float(analysis["performance_score"]),
            model_weight=model_weight,
            strategy_weight=strategy_weight,
            confidence=validation["ai_confidence_index"] / 100,
        )
        state.record("v16_10_recalibration", recalibration)

        prediction = predictive_alignment(
            model_weight=float(recalibration["new_model_weight"]),
            strategy_weight=float(recalibration["new_strategy_weight"]),
            confidence=float(recalibration["new_confidence"]),
            performance=float(analysis["performance_score"]),
        )
        state.record("v16_11_prediction", prediction)

        forecast = scenario_forecast(
            prediction_score=float(prediction["prediction_score"]),
            confidence=float(recalibration["new_confidence"]),
        )
        state.record("v16_12_forecast", forecast)

        optimal = select_optimal_action(
            best_scenario=forecast["best_scenario"],
            confidence=float(recalibration["new_confidence"]),
            risk_level=float(risk["risk_level"]),
            reward_score=float(analysis["performance_score"]),
        )
        state.record("v16_13_optimal_action", optimal)

        orchestrator = orchestrate_execution(
            action=optimal["optimal_action"],
            action_score=float(optimal["action_score"]),
            execution_ready=execution_ready,
            risk_safe=risk["status"] == "SAFE",
        )
        state.record("v16_14_orchestrator", orchestrator)

        monitor = monitor_execution(
            execution_mode=orchestrator["execution_mode"],
            orchestration_score=float(orchestrator["orchestration_score"]),
            health_score=max(0.0, min(float(runtime_health), 1.0)),
            latency_ms=max(0, int(latency_ms)),
        )
        state.record("v16_15_monitor", monitor)

        loop = autonomous_loop(
            system_ready=monitor["system_state"] == "HEALTHY",
            monitor_score=float(monitor["monitor_score"]),
            decision=orchestrator["execution_mode"],
            cycle_id=cycle_id,
        )
        state.record("v16_16_loop", loop)

        state.status = "READY" if not state.errors else "PARTIAL"
        return state.summary()

    except Exception as exc:
        state.record_error("integrated_cycle", exc)
        state.status = "FAILED"
        return state.summary()
