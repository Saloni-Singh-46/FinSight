"""About & Methodology: Multi-Agent Architecture, Evaluation Philosophy, Limitations, and Ethics."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

st.set_page_config(page_title="About & Methodology | FinSight", page_icon="ℹ️", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("ℹ️ FinSight System Architecture & Methodology")

tab1, tab2, tab3 = st.tabs(
    ["Architecture & Agent Flow", "Modeling Methodology", "Limitations & Ethics"]
)

with tab1:
    st.markdown("### 🤖 Multi-Agent LangGraph Orchestration Flow")
    st.markdown("""
    ```mermaid
    graph TD
        UserQuery[User Research Query] --> Planner[Planner Agent: Intent & Plan]
        Planner --> DataAgent[Data Agent: Prices, Fundamentals, Peers]
        Planner --> QuantAgent[Quant Agent: XGBoost, LSTM Fan, SHAP]
        Planner --> SentimentAgent[Sentiment Agent: FinBERT Scoring]
        Planner --> RAGAgent[Filings-RAG Agent: Hybrid Retrieval & Citations]
        
        DataAgent --> Critic[Critic Agent: Consistency & Fact Check]
        QuantAgent --> Critic
        SentimentAgent --> Critic
        RAGAgent --> Critic
        
        Critic -- Pass --> Writer[Writer Agent: 13-Section Research Memo]
        Critic -- Inconsistent / Uncited Claim (Max 2 Rev) --> QuantAgent
        
        Writer --> FinalMemo[Final Memorandum + Plotly Charts + PDF/HTML Export]
    ```
    """)
    st.markdown("""
    - **Stateful Graph (`GraphState`):** Shared immutable Pydantic schema passing structured outputs between agents.
    - **Critic Reflection Loop:** Automatically detects numerical discrepancies or uncited assertions and triggers targeted revisions.
    - **Evidence Store:** Every factual corporate claim is tied to verifiable chunk IDs and page numbers.
    """)

with tab2:
    st.markdown("### 🔬 Quantitative & ML/DL Modeling Protocols")
    st.markdown("""
    - **Tabular Direction Classifier:** XGBoost and LightGBM models trained on 20+ technical indicators, lagged returns, and fundamental balance sheet ratios using 5-fold walk-forward time-series CV with a 5-day purge gap and 2% embargo buffer. Benchmark: Logistic Regression & Naive Persistence.
    - **Deep Learning Time-Series Forecaster:** PyTorch LSTM with Pinball Quantile Loss ($p10, p50, p90$) to generate calibrated 12-month forward price fan charts.
    - **Financial NLP Sentiment:** Fine-tuned `ProsusAI/finbert` model evaluating financial news headlines with volume weighting and exponential recency decay.
    - **Hybrid RAG Retriever:** Blends BM25 lexical token matching with dense semantic embeddings (`all-MiniLM-L6-v2`) and cross-encoder reranking over annual reports and SEC/NSE filings.
    """)

with tab3:
    st.markdown("### ⚖️ Limitations, Risk Disclaimers & Ethics")
    st.markdown("""
    - **Historical Backtesting Limitations:** Historical strategy backtests are simulated out-of-sample representations. Past performance is not indicative of future market returns.
    - **Market Regimes & Black Swans:** Statistical models and neural networks rely on historical distribution patterns and cannot anticipate sudden macroeconomic shocks or geopolitical crises.
    - **AI Hallucination Safeguards:** While FinSight implements multi-agent self-critique loops and citation grounding, outputs must be reviewed by qualified human analysts.
    - **Non-Financial Advice Disclaimer:** FinSight is created strictly for educational, research, and portfolio demonstration purposes. It does not provide personalized investment advice.
    """)
