"""Pydantic Schemas for FinSight REST & SSE API."""

from typing import Any

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    query: str = Field(
        default="Analyze Tata Motors for the next 12 months",
        description="Natural language equity research prompt",
        max_length=500,
    )
    horizon_months: int | None = Field(default=12, ge=1, le=36)
    llm_provider: str | None = Field(
        default="mock", description="'mock' | 'openai' | 'anthropic' | 'ollama'"
    )
    enable_deep_learning: bool = Field(default=True)
    enable_rag: bool = Field(default=True)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "QUEUED", "RUNNING", "COMPLETED", "FAILED"
    created_at: str
    query: str
    company: str
    ticker: str
    message: str


class AnalysisResultResponse(BaseModel):
    job_id: str
    status: str
    query: str
    company: str
    ticker: str
    horizon_months: int
    outlook: str
    confidence: str
    final_memo_markdown: str
    trace_events: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    charts: dict[str, Any]
    critic_audit: dict[str, Any]
    structured_data: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    offline_mode: bool
    components: dict[str, str]
