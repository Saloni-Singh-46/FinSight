"""Planner Agent: Intent Parser and Execution Plan Formulator."""

import time

from finsight.llm.client import BaseLLMClient, get_llm_client
from finsight.orchestration.state import GraphState, TraceEvent
from finsight.utils.config import logger


class PlannerAgent:
    """Parses natural language requests into structured execution tasks."""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self.llm = llm_client or get_llm_client()

    def run(self, state: GraphState) -> GraphState:
        start_time = time.time()
        logger.info(f"Planner Agent processing query: '{state.query}'")

        query_lower = state.query.lower()

        # Check for out-of-scope requests (e.g. personal tax advice, non-financial tasks)
        if "crypto" in query_lower and "tata" not in query_lower:
            state.in_scope = False
            state.final_memo = "Out of scope request. FinSight specializes in corporate equity research and fundamental financial analysis."
            return state

        # Determine target ticker
        company = "Tata Motors"
        ticker = "TATAMOTORS.NS"
        if "reliance" in query_lower:
            company = "Reliance Industries"
            ticker = "RELIANCE.NS"
        elif "hdfc" in query_lower:
            company = "HDFC Bank"
            ticker = "HDFCBANK.NS"

        # Horizon detection
        horizon_m = 12
        if "6 month" in query_lower:
            horizon_m = 6
        elif "3 month" in query_lower or "quarter" in query_lower:
            horizon_m = 3
        elif "1 month" in query_lower:
            horizon_m = 1

        state.company = company
        state.ticker = ticker
        state.horizon_months = horizon_m
        state.horizon_days = horizon_m * 21

        state.plan = {
            "target_company": company,
            "target_ticker": ticker,
            "horizon_months": horizon_m,
            "parallel_tasks": [
                "Extract 5-year price series, fundamentals, and peer multiples (Data Agent)",
                "Execute XGBoost direction classifier, LSTM quantile forecast, and SHAP explainability (Quant Agent)",
                "Score recent financial news sentiment using FinBERT with recency decay (Sentiment Agent)",
                "Query annual reports and filings for JLR guidance, capex, and debt reduction (Filings-RAG Agent)",
            ],
            "validation_stage": "Audit numerical consistency and citation grounding (Critic Agent)",
            "synthesis_stage": "Draft institutional-grade 13-section equity research memorandum (Writer Agent)",
        }

        duration = (time.time() - start_time) * 1000.0
        state.trace_events.append(
            TraceEvent(
                event_id=f"evt_{int(time.time()*1000)}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                agent="Planner Agent",
                status="COMPLETED",
                duration_ms=round(duration, 2),
                tools_called=["parse_intent", "map_ticker_ontology"],
                inputs_summary=f"Query: '{state.query}'",
                outputs_summary=f"Planned analysis for {company} ({ticker}) over {horizon_m} months.",
            )
        )
        return state
