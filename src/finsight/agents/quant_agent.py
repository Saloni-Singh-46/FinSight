"""Quant Agent: Machine Learning Directional Models, LSTM Forecaster, SHAP, and Anomalies."""

import time

from finsight.orchestration.state import GraphState, TraceEvent
from finsight.tools.financial_tools import FinSightToolRegistry
from finsight.utils.config import logger


class QuantAgent:
    """Executes quantitative forecasts, direction classifiers, explainability, and risk health."""

    def __init__(self, tools: FinSightToolRegistry | None = None):
        self.tools = tools or FinSightToolRegistry()

    def run(self, state: GraphState) -> GraphState:
        start_time = time.time()
        logger.info(f"Quant Agent executing models for {state.ticker}...")

        quant_results = self.tools.run_quant_models(state.ticker, horizon_days=state.horizon_days)
        state.quant_agent_output = quant_results

        dir_cls = quant_results["direction_classifier"]
        top_driver = (
            quant_results["shap_top_drivers"][0]
            if quant_results["shap_top_drivers"]
            else {"feature": "N/A"}
        )

        duration = (time.time() - start_time) * 1000.0
        state.trace_events.append(
            TraceEvent(
                event_id=f"evt_{int(time.time()*1000)}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                agent="Quant Agent",
                status="COMPLETED",
                duration_ms=round(duration, 2),
                tools_called=[
                    "xgboost_direction_classifier",
                    "lstm_quantile_fan",
                    "shap_explainer",
                    "anomaly_detector",
                    "credit_scorecard",
                ],
                inputs_summary=f"Horizon: {state.horizon_days}d | Features: 20 technical + fundamental ratios",
                outputs_summary=(
                    f"Direction: {dir_cls['signal']} (Prob Up: {dir_cls['probability_up']*100:.1f}%), "
                    f"Top SHAP driver: {top_driver.get('feature')}, "
                    f"Composite Solvency Health: {quant_results['financial_health']['composite_health_score']}/100"
                ),
            )
        )
        return state
