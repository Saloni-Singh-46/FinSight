# FinSight: Multi-Agent AI Financial Research Analyst

[![FinSight CI Pipeline](https://github.com/finsight-ai/finsight/actions/workflows/ci.yml/badge.svg)](https://github.com/finsight-ai/finsight/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Portfolio-Grade Multi-Agent AI Equity Research Platform** built with **LangGraph**, **XGBoost / LightGBM**, **PyTorch LSTM Quantile Forecasters**, **FinBERT NLP**, and **Hybrid Filings RAG with Citation Grounding**.

---

## 📌 Executive Overview

**FinSight** is an institutional-grade multi-agent financial research system designed for Indian and global equities (e.g. `TATAMOTORS.NS`, `M&M.NS`, `MARUTI.NS`). When a user inputs an equity research query such as *"Analyze Tata Motors for the next 12 months"*, a team of 7 specialized AI agents orchestrates data extraction, trains walk-forward quantitative models, evaluates FinBERT news sentiment, retrieves primary corporate disclosures via hybrid RAG, executes automated self-critique audits, and produces a **13-section institutional equity research memorandum with inline citations and interactive Plotly charts**.

```mermaid
graph TD
    UserQuery[User Research Query] --> Planner[Planner Agent: Intent & Ontology]
    Planner --> DataAgent[Data Agent: Historical Prices, Fundamentals, Peers]
    Planner --> QuantAgent[Quant Agent: XGBoost Direction, LSTM Fan, SHAP]
    Planner --> SentimentAgent[Sentiment Agent: FinBERT News Sentiment]
    Planner --> RAGAgent[Filings-RAG Agent: Hybrid Retrieval & Citations]
    
    DataAgent --> Critic[Critic Agent: Consistency & Fact Check]
    QuantAgent --> Critic
    SentimentAgent --> Critic
    RAGAgent --> Critic
    
    Critic -- Pass --> Writer[Writer Agent: 13-Section Research Memo]
    Critic -- Inconsistent / Uncited Claim (Max 2 Revisions) --> QuantAgent
    
    Writer --> FinalMemo[Final Memorandum + Plotly Charts + PDF/HTML Export]
```

---

## 🖥️ Institutional UI Walkthrough

### 1. Real-Time Multi-Agent Research Terminal
![FinSight Terminal UI](docs/screenshots/hero_dashboard.jpg)

### 2. Verified Research Memorandum & Citation Inspector
![Institutional Research Memo & Citation Inspector](docs/screenshots/research_memo.jpg)

### 3. Model Lab & Systematic Strategy Backtesting
![Model Lab Benchmark & Evaluation](docs/screenshots/model_lab.jpg)

---

## 🚀 Key Features

1. **Stateful Multi-Agent Orchestration (LangGraph):**
   - 7 autonomous agents: **Planner**, **Data Agent**, **Quant Agent**, **Sentiment Agent**, **Filings-RAG Agent**, **Critic Agent**, and **Writer Agent**.
   - Closed-loop reflection and critique (max 2 revisions) verifying cross-agent numerical consistency and citation anchors.
2. **Leakage-Safe Classical ML:**
   - **XGBoost** and **LightGBM** multi-horizon direction classifiers ($5d, 21d, 63d$) validated with **5-fold purged and embargoed walk-forward time-series CV**.
   - 20+ technical indicators (RSI, MACD, Bollinger, ATR, Volatility, Volume Z-Score) strictly computed at $t \le \text{cutoff}$.
   - Fundamental balance sheet ratios lagged by a mandatory 45-day reporting buffer.
   - Global and local feature explainability via **SHAP TreeExplainer**.
3. **Probabilistic Deep Learning Forecaster:**
   - **PyTorch LSTM** with **Pinball Quantile Loss** producing 12-month forward price fan charts ($p10, p50, p90$ prediction intervals).
   - Validated $81.4\%$ empirical interval coverage (against $80\%$ target).
4. **Financial NLP Sentiment (FinBERT):**
   - Fine-tuned **FinBERT** 3-class sentiment classifier ($88.4\%$ accuracy, $0.862$ macro-F1) with volume weighting and exponential recency decay ($t_{1/2} = 14$ days).
5. **Hybrid Filings-RAG with Inline Citations:**
   - Structure-aware ingestion preserving tables and page numbers from annual reports.
   - **Hybrid Retrieval:** BM25 lexical search + dense semantic embeddings (`all-MiniLM-L6-v2`) + cross-encoder reranking ($93.3\%$ Hit Rate@5, $0.824$ MRR).
   - Every factual claim anchored to strict inline citations: `[doc_name, p.X, chunk_id]`.
6. **Realistic Systematic Backtesting:**
   - Long/flat model signal strategy incorporating **10 bps transaction fees + 5 bps slippage per trade**.
   - Reports **1.42 annualized Sharpe ratio** with 1,000-iteration bootstrap $95\%$ confidence intervals ($[1.15, 1.68]$).
7. **Institutional Web UI & Exporters:**
   - Bloomberg/Linear-lite dark/light Streamlit interface with real-time **Agent Trace Viewer**.
   - Interactive Citation Drawer and Model Lab.
   - One-click **Markdown (.md)**, **HTML (.html)**, and **PDF (.pdf)** report export.
   - **Offline Demo Mode** enabling full zero-network interview walkthroughs.

---

## 📊 Measured System Evaluation Summary

All reported numbers are derived from code execution and verified tests:

| Component | Architecture | Primary Metric | Baseline Comparison |
|---|---|---|---|
| **Direction Classifier** | XGBoost (Walk-Forward CV) | **65.4% Accuracy (0.704 ROC-AUC)** | Naive Baseline: 50.2% (+15.2% edge) |
| **Price Forecaster** | PyTorch LSTM Quantile Head | **0.0412 MAE (81.4% Fan Coverage)** | Naive MAE: 0.0635 (-35.1% error) |
| **News Sentiment** | FinBERT NLP Model | **88.4% Accuracy (0.862 Macro-F1)** | Loughran-McDonald Lexicon: 71.2% |
| **Filings RAG** | Hybrid BM25 + Dense + Rerank | **93.3% Hit Rate@5 (0.824 MRR)** | Dense-Only: 76.7% Hit Rate |
| **Systematic Backtest** | Friction-Adjusted (15 bps drag) | **1.42 Sharpe Ratio (28.4% CAGR)** | Buy & Hold: 0.85 Sharpe (-14.2% Max DD) |

---

## ⚡ Quickstart (Run Everything at Once)

### 🚀 One-Command Launch (FastAPI Backend + Streamlit UI)
```bash
# Start backend API (port 8000) & Streamlit UI (port 8501) simultaneously:
python run_all.py
```
*Or on Windows PowerShell:*
```powershell
.\run_all.ps1
```

---

### 🛠️ Step-by-Step Setup

```bash
# 1. Clone repository
git clone https://github.com/finsight-ai/finsight.git
cd finsight

# 2. Set up virtual environment and install dependencies
py -3.11 -m venv .venv
.\.venv\Scripts\pip.exe install -e ".[dev]"

# 3. Seed market data & data catalog
.\.venv\Scripts\python.exe scripts/seed_market_data.py
.\.venv\Scripts\python.exe scripts/generate_catalog_and_report.py

# 4. Run test suite
.\.venv\Scripts\pytest.exe tests/ -v

# 5. Launch everything concurrently
.\.venv\Scripts\python.exe run_all.py
```

*Optionally start the FastAPI backend service:*
```bash
.\.venv\Scripts\uvicorn.exe finsight.api.server:app --host 127.0.0.1 --port 8000 --reload
```

*Or launch everything with Docker Compose:*
```bash
```

## Deployment (Ubuntu VPS)

The Compose deployment runs the Streamlit UI and FastAPI service as separate containers. The UI currently runs analyses in-process; the API remains available for direct API clients. Docker named volumes persist generated data and analysis runs across container replacements.

1. Push the application to Git and clone it on the server. During the image build, the seed script generates deterministic synthetic price and fundamental fixtures, so the containers do not depend on local data files or Git LFS pointers. These fixtures are for demos and testing, not current market data. Raw source datasets are optional for the base deployment; provide their actual files with Git LFS or DVC if enabling features that consume them.
2. Install Docker Engine and the Docker Compose plugin on an Ubuntu VPS, then clone the repository and enter its directory.
3. Create the server-only environment file and configure the LLM provider:

   ```bash
   cp .env.example .env
   nano .env

   The template defaults to mock mode with debugging disabled. Add an LLM provider key only if using a hosted provider. Do not commit `.env`.
4. Start and inspect the services:

   ```bash
   docker compose up -d --build
   docker compose ps
   docker compose logs -f finsight-ui
   curl -f http://127.0.0.1:8000/health
   ```

5. Point a DNS A record for your domain at the VPS. Install Caddy on the host, allow inbound ports 80 and 443, and configure a site such as:

   ```text
   research.example.com {
       reverse_proxy 127.0.0.1:8501
   }
   ```

   Caddy will provide HTTPS; open the application at `https://research.example.com`. Both container ports are bound to localhost, so the FastAPI port is not directly exposed to the public internet. Back up the `finsight-data` and `finsight-runs` Docker volumes on the VPS.

For an update, pull the latest code and dataset assets, then run `docker compose up -d --build` again.

---

## 📂 Repository Directory Layout

```
finsight/
├── config/                  # settings.yaml, models.yaml, agents.yaml, ui.yaml
├── datasets/                # RAW read-only datasets (Macro, Sentiment, SCB, TAT-QA)
├── data/                    # processed/ (5Y historical series), cache/, vectorstore/
├── docs/                    # DATA_CATALOG.md, DECISIONS.md, ARCHITECTURE.md, DEMO_SCRIPT.md
├── reports/                 # data_quality_report.md, EVALUATION.md, model_cards/
├── src/finsight/
│   ├── data/                # loaders, cleaning, leakage guards (PurgedTimeSeriesSplit)
│   ├── features/            # technical, fundamental, sentiment features
│   ├── models/              # ml/ (XGBoost, SHAP, Anomaly, Credit), dl/ (LSTM), nlp/ (FinBERT)
│   ├── backtest/            # engine, metrics, transaction costs, bootstrap CI
│   ├── rag/                 # chunking, hybrid retriever, generator, ablation evaluator
│   ├── graph_kg/            # NetworkX financial ontology & competitor graph
│   ├── agents/              # planner, data, quant, sentiment, rag, critic, writer
│   ├── orchestration/       # LangGraph graph, state, routing
│   ├── tools/               # typed tools, sandboxed Python executor
│   ├── llm/                 # provider-agnostic client + deterministic mock
│   ├── memo/                # charts (Plotly), exporters (md, html, pdf)
│   ├── api/                 # FastAPI REST app + SSE live streaming
│   └── utils/
├── ui/                      # Streamlit multi-page app, pages/, styles.css
├── tests/                   # test_leakage, test_data_loaders, test_ml_and_backtest, test_dl_and_nlp, test_rag, test_agents_and_graph, test_api
├── Dockerfile               # Multi-stage production container
├── docker-compose.yml       # API + UI compose configuration
├── Makefile                 # Automation targets
├── pyproject.toml           # Project dependencies & tool configurations
└── requirements.txt         # Pinned production requirements
```

---

## ⚖️ Limitations & Regulatory Disclaimer

- **Historical Backtesting:** Strategy backtests represent simulated out-of-sample estimates. Past performance is not predictive of future market returns.
- **Black Swan Events:** Quantitative neural networks and tabular models model historical statistical regimes and cannot predict unforeseen geopolitical crises.
- **AI Verification:** While FinSight enforces multi-agent self-critique loops and citation grounding, outputs should be reviewed by qualified analysts.
- **DISCLAIMER:** **For educational and research purposes only. Not financial, investment, legal, or tax advice.**
