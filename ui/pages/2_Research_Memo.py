"""Institutional Equity Research Memorandum View & Citation Drawer."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from finsight.memo.exporter import MemoExporter
from finsight.orchestration.graph import FinSightOrchestrator
from finsight.orchestration.state import GraphState
from finsight.tools.financial_tools import FinSightToolRegistry

st.set_page_config(page_title="Research Memo | FinSight", page_icon="📑", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📑 Institutional Research Memorandum")

if "current_state" not in st.session_state or st.session_state.current_state is None:
    with st.spinner("Generating institutional memorandum for Tata Motors..."):
        tool_reg = FinSightToolRegistry()
        orchestrator = FinSightOrchestrator(tools=tool_reg)
        st.session_state.current_state = orchestrator.execute(
            "Analyze Tata Motors for the next 12 months"
        )

state: GraphState = st.session_state.current_state
exporter = MemoExporter()

# Export Download Action Bar
c_exp1, c_exp2, c_exp3, c_exp4 = st.columns([1.5, 1, 1, 1])
with c_exp1:
    st.markdown(f"**Target Asset:** `{state.company} ({state.ticker})`")

with c_exp2:
    st.download_button(
        label="📥 Download Markdown (.md)",
        data=state.final_memo,
        file_name=f"{state.ticker}_FinSight_Memo.md",
        mime="text/markdown",
        use_container_width=True,
    )

with c_exp3:
    html_path = exporter.export_html(state.final_memo, state.job_id, company=state.company)
    with open(html_path, "r", encoding="utf-8") as f:
        html_data = f.read()
    st.download_button(
        label="🌐 Download HTML (.html)",
        data=html_data,
        file_name=f"{state.ticker}_FinSight_Memo.html",
        mime="text/html",
        use_container_width=True,
    )

with c_exp4:
    pdf_path = exporter.export_pdf(state.final_memo, state.job_id, company=state.company)
    if pdf_path.endswith(".pdf") and Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        st.download_button(
            label="📄 Download PDF (.pdf)",
            data=pdf_data,
            file_name=f"{state.ticker}_FinSight_Memo.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

st.markdown("---")

# Layout: Main Memo + Sidebar Citation Drawer
col_memo, col_drawer = st.columns([2.2, 1])

with col_memo:
    st.markdown(state.final_memo)

with col_drawer:
    st.markdown("### 🔍 Verified Primary Source Drawer")
    st.markdown("Every factual claim in the memo is anchored to verified corporate filings.")

    if state.citations:
        for c in state.citations:
            with st.expander(f"📌 {c.get('cite_key')}", expanded=True):
                st.markdown(f"**Document:** `{c.get('doc_name')}`")
                st.markdown(
                    f"**Page:** `{c.get('page_number')}` | **Section:** `{c.get('section')}`"
                )
                st.markdown(f"**Original Excerpt:**\n> {c.get('full_text')}")
    else:
        st.info("No citation anchors in current session.")
