"""Sentiment Agent: Market & Media Sentiment Evaluation with FinBERT."""

import time

from finsight.orchestration.state import GraphState, TraceEvent
from finsight.tools.financial_tools import FinSightToolRegistry
from finsight.utils.config import logger


class SentimentAgent:
    """Evaluates news sentiment, calculates volume-weighted scores, and tracks recency trends."""

    def __init__(self, tools: FinSightToolRegistry | None = None):
        self.tools = tools or FinSightToolRegistry()

    def run(self, state: GraphState) -> GraphState:
        start_time = time.time()
        logger.info(f"Sentiment Agent evaluating market headlines for {state.company}...")

        sent_results = self.tools.analyze_sentiment(state.company)
        state.sentiment_agent_output = sent_results

        duration = (time.time() - start_time) * 1000.0
        state.trace_events.append(
            TraceEvent(
                event_id=f"evt_{int(time.time()*1000)}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                agent="Sentiment Agent",
                status="COMPLETED",
                duration_ms=round(duration, 2),
                tools_called=["finbert_headline_scorer", "recency_decay_aggregator"],
                inputs_summary=f"Analyzed {sent_results['total_articles_scored']} recent financial headlines",
                outputs_summary=(
                    f"Overall Sentiment: {sent_results['overall_sentiment']}, "
                    f"Net Score: {sent_results['net_sentiment_score']}, "
                    f"Positive/Negative count: {sent_results['positive_count']}/{sent_results['negative_count']}"
                ),
            )
        )
        return state
