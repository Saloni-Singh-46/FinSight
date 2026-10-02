"""Data Agent: Financial Data Extraction and Quality Validation."""

import time

from finsight.orchestration.state import GraphState, TraceEvent
from finsight.tools.financial_tools import FinSightToolRegistry
from finsight.utils.config import logger


class DataAgent:
    """Extracts market history, fundamental ratios, balance sheet items, and peer metrics."""

    def __init__(self, tools: FinSightToolRegistry | None = None):
        self.tools = tools or FinSightToolRegistry()

    def run(self, state: GraphState) -> GraphState:
        start_time = time.time()
        logger.info(f"Data Agent fetching data for {state.ticker}...")

        snapshot = self.tools.load_price_and_fundamentals(state.ticker)
        state.data_agent_output = snapshot

        duration = (time.time() - start_time) * 1000.0
        state.trace_events.append(
            TraceEvent(
                event_id=f"evt_{int(time.time()*1000)}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                agent="Data Agent",
                status="COMPLETED",
                duration_ms=round(duration, 2),
                tools_called=["load_price_history", "load_fundamentals", "load_peer_multiples"],
                inputs_summary=f"Target: {state.ticker}",
                outputs_summary=(
                    f"Extracted latest price ₹{snapshot['latest_price']}, "
                    f"P/E {snapshot['fundamentals'].get('pe_ratio', 'N/A')}x, "
                    f"ROE {snapshot['fundamentals'].get('roe_pct', 'N/A')}%, "
                    f"{len(snapshot['peers'])} peers loaded."
                ),
            )
        )
        return state
