# FinSight System Architecture & Engineering Design

```mermaid
graph TD
    subgraph ClientLayer [Client & Presentation Layer]
        UI[Streamlit Multi-Page Dashboard]
        SSE[SSE Live Agent Trace Streamer]
        PDF[ReportLab / HTML Export Engine]
    end

    subgraph APILayer [FastAPI Backend Service]
        AnalyzeEndpoint["POST /analyze"]
        StreamEndpoint["GET /jobs/{id}/stream"]
        ResultEndpoint["GET /jobs/{id}"]
        EvalEndpoint["GET /evaluation/summary"]
    end

    subgraph AgentOrchestration [LangGraph Multi-Agent Engine]
        Planner[Planner Agent]
        DataAgent[Data Agent]
        QuantAgent[Quant Agent]
        SentimentAgent[Sentiment Agent]
        RAGAgent[Filings-RAG Agent]
        Critic[Critic Agent]
        Writer[Writer Agent]
    end

    subgraph ModelLayer [Quantitative & NLP Models]
        XGBoost[XGBoost Direction Classifier]
        LSTM[PyTorch LSTM Multi-Quantile Forecaster]
        SHAP[SHAP TreeExplainer & Local Drivers]
        Anomaly[Isolation Forest & Rolling MAD]
        Credit[Credit Risk Scorecard]
        FinBERT[FinBERT Sentiment Classifier]
        RAG[Hybrid BM25 + Dense all-MiniLM + Cross-Encoder]
    end

    subgraph DataLayer [Storage & Data Infrastructure]
        ProcessedCache[(data/processed/ 5Y Historical Data)]
        VectorStore[(ChromaDB / In-Memory Embeddings)]
        RawDatasets[(Datasets/ Raw Read-Only)]
        TraceLogs[(runs/{id}/trace.json)]
    end

    UI --> AnalyzeEndpoint
    UI --> StreamEndpoint
    UI --> ResultEndpoint
    
    AnalyzeEndpoint --> Planner
    Planner --> DataAgent
    Planner --> QuantAgent
    Planner --> SentimentAgent
    Planner --> RAGAgent

    DataAgent --> ProcessedCache
    QuantAgent --> XGBoost
    QuantAgent --> LSTM
    QuantAgent --> SHAP
    QuantAgent --> Anomaly
    QuantAgent --> Credit
    SentimentAgent --> FinBERT
    RAGAgent --> RAG
    RAG --> VectorStore

    DataAgent --> Critic
    QuantAgent --> Critic
    SentimentAgent --> Critic
    RAGAgent --> Critic

    Critic -- Pass --> Writer
    Critic -- Inconsistency / Uncited Assertion (Max 2 Rev) --> QuantAgent
    Writer --> PDF
    Writer --> TraceLogs
```

---

## 1. Core Architectural Pillars

### A. Stateful Orchestration with LangGraph
- **`GraphState` Schema:** Strongly typed immutable Pydantic model passing data snapshots, model predictions, citations, evidence dictionaries, and execution traces.
- **Reflection Loop:** Critic Agent audits numbers, cross-agent consistency, and source citations. If discrepancies or ungrounded assertions are detected, the graph loops back with actionable feedback (hard cap: 2 revisions).

### B. Offline-First Reproducibility
- Operates deterministically from local cached 5-year daily market histories, financial statements, and pre-indexed annual reports.
- Live external network connections (e.g. yfinance, live news RSS) exist as optional enrichments behind feature flags.

### C. Zero Look-Ahead Leakage Protocol
- Walk-forward time-series splits with a 5-day purge gap and 2% embargo buffer.
- Technical indicators consume data strictly $\le t$.
- Fundamental ratios lagged by 45 days.
- Automated tests in `tests/test_leakage.py` assert bit-for-bit feature invariance when future prices are modified.
