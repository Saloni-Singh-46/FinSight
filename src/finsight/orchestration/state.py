"""Typed LangGraph GraphState and TraceEvent Models for FinSight."""

import time
from typing import Any

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    event_id: str
    timestamp: str
    agent: str
    status: str  # "QUEUED", "RUNNING", "COMPLETED", "REVISED", "FAILED"
    duration_ms: float = 0.0
    tools_called: list[str] = Field(default_factory=list)
    inputs_summary: str = ""
    outputs_summary: str = ""
    critic_feedback: str | None = None


class GraphState(BaseModel):
    job_id: str = Field(default_factory=lambda: f"job_{int(time.time()*1000)}")
    query: str = "Analyze Tata Motors for the next 12 months"
    company: str = "Tata Motors"
    ticker: str = "TATAMOTORS.NS"
    horizon_months: int = 12
    horizon_days: int = 252
    intent: str = "equity_research"
    in_scope: bool = True

    # Agent structured outputs
    plan: dict[str, Any] = Field(default_factory=dict)
    data_agent_output: dict[str, Any] = Field(default_factory=dict)
    quant_agent_output: dict[str, Any] = Field(default_factory=dict)
    sentiment_agent_output: dict[str, Any] = Field(default_factory=dict)
    rag_agent_output: dict[str, Any] = Field(default_factory=dict)

    # Evidence & Verification
    evidence_store: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    critic_output: dict[str, Any] = Field(default_factory=dict)
    revision_count: int = 0
    max_revisions: int = 2

    # Final Synthesis
    final_memo: str = ""
    final_memo_structured: dict[str, Any] = Field(default_factory=dict)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    status: str = "INITIALIZED"
