"""Filings-RAG Agent: Grounded Corporate Filings and Transcripts Ingestion & Retrieval."""

import time

from finsight.orchestration.state import GraphState, TraceEvent
from finsight.tools.financial_tools import FinSightToolRegistry
from finsight.utils.config import logger


class FilingsRAGAgent:
    """Queries corporate annual reports, 10-K filings, and transcripts with strict inline citations."""

    def __init__(self, tools: FinSightToolRegistry | None = None):
        self.tools = tools or FinSightToolRegistry()

    def run(self, state: GraphState) -> GraphState:
        start_time = time.time()
        logger.info(f"Filings-RAG Agent querying corporate disclosures for {state.company}...")

        query = f"Financial performance, JLR EBIT margin guidance, debt reduction, EV market share, and key risk factors for {state.company}"
        rag_res = self.tools.search_filings_rag(query, company=state.company)

        state.rag_agent_output = rag_res
        state.citations.extend(rag_res.get("citations", []))
        state.evidence_store.update(rag_res.get("evidence_store", {}))

        duration = (time.time() - start_time) * 1000.0
        state.trace_events.append(
            TraceEvent(
                event_id=f"evt_{int(time.time()*1000)}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                agent="Filings-RAG Agent",
                status="COMPLETED",
                duration_ms=round(duration, 2),
                tools_called=[
                    "hybrid_bm25_dense_retriever",
                    "cross_encoder_reranker",
                    "citation_grounder",
                ],
                inputs_summary=f"Query: '{query[:60]}...'",
                outputs_summary=(
                    f"Retrieved and grounded {len(rag_res.get('citations', []))} verified passages from {rag_res.get('top_source', 'Filings')}."
                ),
            )
        )
        return state
