"""Live Multi-Agent Analysis and Agent Trace Inspection Page."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from finsight.orchestration.graph import FinSightOrchestrator
from finsight.orchestration.state import GraphState
from finsight.tools.financial_tools import FinSightToolRegistry

st.set_page_config(page_title="Live Analysis | FinSight", page_icon="⚡", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("⚡ Live Multi-Agent Execution & Trace Inspector")
st.markdown(
    "Inspect real-time execution states, inter-agent message payloads, tool invocations, and self-critique loops."
)

if "current_state" not in st.session_state or st.session_state.current_state is None:
    # Auto-initialize a default run
    with st.spinner("Initializing FinSight multi-agent run for Tata Motors..."):
        tool_reg = FinSightToolRegistry()
        orchestrator = FinSightOrchestrator(tools=tool_reg)
        st.session_state.current_state = orchestrator.execute(
            "Analyze Tata Motors for the next 12 months"
        )

state: GraphState = st.session_state.current_state

tab1, tab2, tab3 = st.tabs(
    ["Timeline Trace", "Agent State Inspector", "Evidence Store & Citations"]
)

with tab1:
    st.markdown("### 🔄 Sequential & Parallel Agent Timeline")
    for idx, evt in enumerate(state.trace_events, 1):
        with st.expander(
            f"Step {idx}: {evt.agent} — [{evt.status}] ({evt.duration_ms:.1f}ms)",
            expanded=(idx in [1, 2, 6]),
        ):
            st.markdown(f"**Timestamp:** `{evt.timestamp}` | **Status:** `{evt.status}`")
            st.markdown(f"**Tools Executed:** `{'`, `'.join(evt.tools_called)}`")
            st.markdown(f"**Input:** {evt.inputs_summary}")
            st.markdown(f"**Output Payload:** {evt.outputs_summary}")
            if evt.critic_feedback:
                st.info(f"**Critic Revision Note:** {evt.critic_feedback}")

with tab2:
    st.markdown("### 🧬 Raw Typed GraphState Snapshot")
    st.json(
        {
            "job_id": state.job_id,
            "company": state.company,
            "ticker": state.ticker,
            "horizon_days": state.horizon_days,
            "revision_count": state.revision_count,
            "data_agent_output": state.data_agent_output,
            "quant_agent_output": {
                k: v for k, v in state.quant_agent_output.items() if k != "price_fan_chart"
            },
            "sentiment_agent_output": state.sentiment_agent_output,
            "critic_audit": state.critic_output,
        }
    )

with tab3:
    st.markdown("### 📚 Grounded Evidence Store & Primary Citations")
    if state.citations:
        for c in state.citations:
            st.markdown(
                f"""
            <div class="metric-card">
                <span class="citation-chip">{c.get('cite_key')}</span>
                <span style="font-size: 12px; color: #9ca3af; margin-left: 10px;">Section: <b>{c.get('section')}</b></span>
                <div style="font-size: 13px; margin-top: 8px; color: #f9fafb;">"{c.get('full_text')}"</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No citations currently loaded.")
