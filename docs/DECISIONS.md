# FinSight Architecture & Engineering Decisions Record (ADR)

## ADR 001: Offline-First Architecture & Data Strategy
- **Status:** Accepted
- **Context:** The prompt mandates an offline-first financial analyst system that operates deterministically on local datasets, with live APIs (e.g., yfinance) treated strictly as optional enrichments behind feature flags.
- **Decision:**
  1. We establish a local processed data cache in `data/processed/` containing high-resolution 5-year historical price, volume, and fundamental financial records for the core target company (**Tata Motors - `TATAMOTORS.NS`**) and key automotive/index peers (**Mahindra & Mahindra, Maruti Suzuki, Ashok Leyland, NIFTY50**).
  2. For credit risk and financial health scoring, we utilize the provided `statlog+german+credit+data` (1,000 corporate/retail credit observations with 20 risk attributes) paired with classic transparent fundamental scorecards (Altman Z-Score and Piotroski F-Score).
  3. For financial sentiment, we leverage the provided `Datasets/sent_train.csv` (9,938 labeled financial headlines/tweets) and `Datasets/train.csv` (financial phrase bank).
  4. For filings and transcripts RAG, we utilize `Datasets/tatqa_dataset_train.json` (2,201 real-world financial table-text contexts from corporate filings) and pre-indexed SEC/NSE annual reports in ChromaDB.
- **Consequences:** Reproducible, zero-network dependency demo mode, completely immune to API outages, rate limits, or expired keys.

## ADR 002: Look-Ahead Bias and Temporal Leakage Safeguards
- **Status:** Accepted
- **Context:** Financial ML models frequently suffer from target leakage and look-ahead bias if features or normalization use future information.
- **Decision:**
  1. All train/test validation splits enforce strict chronological walk-forward / expanding-window time-series CV with a 5-day purge gap and 2% embargo buffer.
  2. Technical indicators at time $t$ strictly consume data strictly available at or prior to $t$.
  3. Fundamental balance sheet ratios are lagged by a mandatory 45-day reporting buffer.
  4. Unit tests in `tests/test_leakage.py` verify that feature calculation functions yield identical values when future rows are appended or modified.
- **Consequences:** Realistic, uninflated backtest results that truthfully reflect out-of-sample edge.

## ADR 003: Model Architecture & Baselines Hierarchy
- **Status:** Accepted
- **Context:** To ensure honesty, all complex models must be benchmarked against transparent statistical baselines.
- **Decision:**
  1. **Tabular Direction Classifier:** XGBoost and LightGBM benchmarked against Logistic Regression and a Naive Directional Persistence baseline.
  2. **Time-Series Forecaster:** PyTorch LSTM with quantile loss (p10, p50, p90) benchmarked against Naive Last-Value, Seasonal Naive, ARIMA, and XGBoost Regressor.
  3. **Sentiment NLP:** Fine-tuned `ProsusAI/finbert` benchmarked against Zero-Shot FinBERT and Loughran-McDonald financial dictionary lexicon baseline.
  4. **Filings RAG:** Hybrid retrieval (Dense `all-MiniLM-L6-v2` + BM25) + Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`) benchmarked against Dense-only and BM25-only retrieval.
- **Consequences:** Complete ablation transparency and defensible model validation.

## ADR 004: Multi-Agent Orchestration & Self-Critique Loop
- **Status:** Accepted
- **Context:** Multi-agent LLM systems can hallucinate, produce inconsistent cross-agent numbers, or fail to ground claims.
- **Decision:**
  1. We use **LangGraph** with a strongly typed Pydantic `GraphState`.
  2. A dedicated **Critic Agent** audits quantitative consistency, checks for contradictions between agents, verifies citation anchors, and triggers a reflection loop (max 2 revisions) if inconsistencies are detected.
  3. All claims in the generated memo are anchored to verified source chunks with strict citation syntax `[doc, p.X, chunk-id]`.
- **Consequences:** Bulletproof claim verification, low hallucination risk, and audit-ready outputs.
