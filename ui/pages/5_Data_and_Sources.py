"""Data & Sources Page: Dataset Catalog, Quality Audits, and Document Library."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

st.set_page_config(page_title="Data & Sources | FinSight", page_icon="📁", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📁 Data Inventory, Quality Audits & Sources")
st.markdown(
    "Transparent overview of all raw, interim, and processed data sources feeding the FinSight intelligence platform."
)

tab1, tab2, tab3 = st.tabs(
    ["Component Data Mapping", "Data Quality & Leakage Audit", "Document Library & Ingestion"]
)

with tab1:
    st.markdown("### 🗺️ Component-to-Dataset Source Mapping")
    mapping_data = [
        {
            "System Component": "Sentiment Agent (FinBERT)",
            "Local Dataset Source": "Datasets/sent_train.csv, Datasets/train.csv",
            "Records": "9,938 headlines/tweets",
            "Role": "Financial sentiment classification & headline scoring",
            "Status": "✅ Clean & Indexed",
        },
        {
            "System Component": "Filings-RAG Agent",
            "Local Dataset Source": "Datasets/tatqa_dataset_train.json, Annual Reports",
            "Records": "2,201 table-text pairs",
            "Role": "Hybrid retrieval, citations & financial QA",
            "Status": "✅ Clean & Indexed",
        },
        {
            "System Component": "Quant Agent (Credit Scorecard)",
            "Local Dataset Source": "Datasets/statlog+german+credit+data/",
            "Records": "1,000 credit records",
            "Role": "Credit risk modeling & financial solvency scorecard",
            "Status": "✅ Trained",
        },
        {
            "System Component": "Quant Agent (Anomaly Detection)",
            "Local Dataset Source": "Datasets/creditcard.csv, Local Time Series",
            "Records": "284,807 events",
            "Role": "Isolation Forest & Rolling MAD regime detection",
            "Status": "✅ Calibrated",
        },
        {
            "System Component": "Classical ML & DL Forecasters",
            "Local Dataset Source": "data/processed/*_daily.csv",
            "Records": "1,500+ daily bars per ticker",
            "Role": "5-year OHLCV, volume, technical indicators, targets",
            "Status": "✅ 0 Nulls, Split-Adjusted",
        },
        {
            "System Component": "Data Agent (Macro / Banking Context)",
            "Local Dataset Source": "Datasets/Basic Statistical Returns of SCBs/",
            "Records": "100+ RBI workbooks",
            "Role": "Banking credit distribution, interest rates & sector trends",
            "Status": "✅ Cataloged",
        },
    ]
    st.dataframe(pd.DataFrame(mapping_data), use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### 🛡️ Look-Ahead Bias & Temporal Leakage Safeguards")
    st.markdown(r"""
    FinSight implements strict institutional data integrity protocols:
    1. **Strict Walk-Forward Cross-Validation:** No shuffling on time series. Folds expand chronologically.
    2. **Purge & Embargo Windows:** 5-day purge gap and 2% embargo buffer to prevent autocorrelation leakage.
    3. **Point-In-Time Features:** All technical indicators (RSI, MACD, Bollinger, ATR) at time $t$ consume data strictly $\le t$.
    4. **45-Day Fundamental Reporting Lag:** Balance sheet and quarterly numbers are lagged by a mandatory 45-day reporting buffer.
    5. **Automated Unit Tests:** `tests/test_leakage.py` asserts bit-for-bit invariance when future prices are modified.
    """)

with tab3:
    st.markdown("### 📄 Corporate Filings Library")
    docs = [
        {
            "Document Name": "TataMotors_AR_FY24.pdf",
            "Company": "Tata Motors Limited",
            "Fiscal Year": "FY24",
            "Pages": "412",
            "Ingestion Chunks": "450+",
            "Vector Status": "Indexed in Hybrid Store",
        },
        {
            "Document Name": "TataMotors_Q4_FY24_Transcript.pdf",
            "Company": "Tata Motors Limited",
            "Fiscal Year": "Q4 FY24",
            "Pages": "24",
            "Ingestion Chunks": "35",
            "Vector Status": "Indexed in Hybrid Store",
        },
        {
            "Document Name": "TAT-QA Financial Benchmark",
            "Company": "Multi-Entity S&P / Global 500",
            "Fiscal Year": "2019-2023",
            "Pages": "N/A",
            "Ingestion Chunks": "2,201 Tables/Texts",
            "Vector Status": "Benchmark Grounding Set",
        },
    ]
    st.dataframe(pd.DataFrame(docs), use_container_width=True, hide_index=True)
