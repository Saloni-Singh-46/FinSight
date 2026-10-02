"""FinSight LangGraph Stateful Orchestrator with Reflection and Revision Loops."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from finsight.agents.critic_agent import CriticAgent
from finsight.agents.data_agent import DataAgent
from finsight.agents.planner import PlannerAgent
from finsight.agents.quant_agent import QuantAgent
from finsight.agents.rag_agent import FilingsRAGAgent
from finsight.agents.sentiment_agent import SentimentAgent
from finsight.agents.writer_agent import WriterAgent
from finsight.orchestration.state import GraphState
from finsight.tools.financial_tools import FinSightToolRegistry


class FinSightOrchestrator:
    """Coordinates stateful multi-agent execution with LangGraph."""

    def __init__(
        self,
        tools: FinSightToolRegistry | None = None,
        on_event_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.tools = tools or FinSightToolRegistry()
        self.callback = on_event_callback

        # Initialize agents
        self.planner = PlannerAgent()
        self.data_agent = DataAgent(self.tools)
        self.quant_agent = QuantAgent(self.tools)
        self.sentiment_agent = SentimentAgent(self.tools)
        self.rag_agent = FilingsRAGAgent(self.tools)
        self.critic_agent = CriticAgent()
        self.writer_agent = WriterAgent()

        self.workflow = self._build_graph()

    def _build_graph(self) -> Any:
        """Builds LangGraph state machine with conditional critic reflection loop."""
        graph = StateGraph(GraphState)

        # 1. Define Nodes
        graph.add_node("planner", self._node_planner)
        graph.add_node("data_agent", self._node_data)
        graph.add_node("quant_agent", self._node_quant)
        graph.add_node("sentiment_agent", self._node_sentiment)
        graph.add_node("rag_agent", self._node_rag)
        graph.add_node("critic", self._node_critic)
        graph.add_node("writer", self._node_writer)

        # 2. Define Flow Edges
        graph.set_entry_point("planner")

        # Planner transitions to parallel execution pipeline
        graph.add_edge("planner", "data_agent")
        graph.add_edge("data_agent", "quant_agent")
        graph.add_edge("quant_agent", "sentiment_agent")
        graph.add_edge("sentiment_agent", "rag_agent")
        graph.add_edge("rag_agent", "critic")

        # Critic conditional routing
        def critic_condition(state: GraphState) -> str:
            if not state.in_scope:
                return "writer"
            status = state.critic_output.get("status", "PASS")
            if status == "REVISE" and state.revision_count <= state.max_revisions:
                return "quant_agent"  # Revision loop
            return "writer"

        graph.add_conditional_edges(
            "critic",
            critic_condition,
            {
                "quant_agent": "quant_agent",
                "writer": "writer",
            },
        )

        graph.add_edge("writer", END)
        return graph.compile()

    def _notify_event(self, state: GraphState):
        if self.callback and state.trace_events:
            last_event = state.trace_events[-1]
            self.callback(
                {
                    "job_id": state.job_id,
                    "agent": last_event.agent,
                    "status": last_event.status,
                    "duration_ms": last_event.duration_ms,
                    "outputs_summary": last_event.outputs_summary,
                    "total_events": len(state.trace_events),
                }
            )

    def _node_planner(self, state: GraphState) -> GraphState:
        s = self.planner.run(state)
        self._notify_event(s)
        return s

    def _node_data(self, state: GraphState) -> GraphState:
        s = self.data_agent.run(state)
        self._notify_event(s)
        return s

    def _node_quant(self, state: GraphState) -> GraphState:
        s = self.quant_agent.run(state)
        self._notify_event(s)
        return s

    def _node_sentiment(self, state: GraphState) -> GraphState:
        s = self.sentiment_agent.run(state)
        self._notify_event(s)
        return s

    def _node_rag(self, state: GraphState) -> GraphState:
        s = self.rag_agent.run(state)
        self._notify_event(s)
        return s

    def _node_critic(self, state: GraphState) -> GraphState:
        s = self.critic_agent.run(state)
        self._notify_event(s)
        return s

    def _node_writer(self, state: GraphState) -> GraphState:
        s = self.writer_agent.run(state)
        self._notify_event(s)
        return s

    def execute(self, query: str = "Analyze Tata Motors for the next 12 months") -> GraphState:
        """Executes full multi-agent graph from query to final research memo."""
        initial_state = GraphState(query=query)
        final_state_dict = self.workflow.invoke(initial_state)

        final_state = GraphState.model_validate(final_state_dict)

        # Save run trace to runs/{id}/trace.json
        run_dir = Path("runs") / final_state.job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "trace.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": final_state.job_id,
                    "query": final_state.query,
                    "company": final_state.company,
                    "ticker": final_state.ticker,
                    "horizon_months": final_state.horizon_months,
                    "events": [e.model_dump() for e in final_state.trace_events],
                    "critic_output": final_state.critic_output,
                    "final_memo_length": len(final_state.final_memo),
                },
                f,
                indent=2,
            )

        return final_state
