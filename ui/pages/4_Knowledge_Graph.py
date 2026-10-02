"""Knowledge Graph Explorer: Companies, Sectors, Peers, Segments, and Macro Supply Chain."""

import sys
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from finsight.graph_kg.knowledge_graph import FinancialKnowledgeGraph

st.set_page_config(page_title="Knowledge Graph | FinSight", page_icon="🕸️", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🕸️ Financial Knowledge Graph & Supply Chain Ontology")
st.markdown(
    "Interactive multi-relational network connecting target companies, peer competitors, business segments, and macro dependencies."
)

kg = FinancialKnowledgeGraph()
graph = kg.graph

# Compute 2D Spring Layout for Plotly
pos = nx.spring_layout(graph, seed=42, k=0.8)

# Edge traces
edge_x = []
edge_y = []
edge_labels = []

for edge in graph.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_labels.append(edge[2].get("relation", ""))

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1.5, color="#374151"),
    hoverinfo="none",
    mode="lines",
)

# Node traces by category
node_x = []
node_y = []
node_text = []
node_color = []
node_size = []

color_map = {
    "Company": "#2563eb",  # Blue
    "Sector": "#8b5cf6",  # Purple
    "Industry": "#06b6d4",  # Cyan
    "Segment": "#10b981",  # Green
    "Macro": "#f59e0b",  # Amber
}

for node in graph.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    data = graph.nodes[node]
    node_type = data.get("type", "Company")
    name = data.get("name", node)
    node_text.append(f"<b>{name}</b><br>Type: {node_type}<br>ID: {node}")
    node_color.append(color_map.get(node_type, "#9ca3af"))
    node_size.append(28 if node_type == "Company" else (22 if node_type == "Segment" else 18))

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    hoverinfo="text",
    text=[graph.nodes[n].get("name", n).split()[0] for n in graph.nodes()],
    textposition="bottom center",
    hovertext=node_text,
    marker=dict(
        color=node_color,
        size=node_size,
        line=dict(width=2, color="#111827"),
    ),
)

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title="<b>FinSight Multi-Relational Knowledge Network</b>",
        titlefont_size=16,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=40),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=580,
    ),
)

col_graph, col_legend = st.columns([3, 1])

with col_graph:
    st.plotly_chart(fig, use_container_width=True)

with col_legend:
    st.markdown("### 🏷️ Node Legend & Types")
    st.markdown("""
    - 🔵 **Company (Primary Target & Peers):** `Tata Motors`, `M&M`, `Maruti Suzuki`, `Ashok Leyland`
    - 🟢 **Segment Units:** `JLR`, `Tata Commercial Vehicles`, `Tata Passenger & EV`
    - 🟣 **Sector / Industry:** `Automotive Sector`, `Commercial Vehicles`
    - 🟠 **Macro Risk Factors:** `Steel Commodities`, `Semiconductors`, `Interest Rates`, `Lithium Cells`
    """)
    st.markdown("---")
    st.markdown("### 🔍 Extracted Graph Insights")
    peers = kg.get_company_peers("TATAMOTORS.NS")
    st.markdown(f"**Identified Competitors ({len(peers)}):**")
    for p in peers:
        st.markdown(f"- **{p['name']}** (`{p['ticker']}`) — *{p['competing_segment']}*")

    risks = kg.get_risk_factors("TATAMOTORS.NS")
    st.markdown(f"**Direct Macro Exposures ({len(risks)}):**")
    for r in risks:
        st.markdown(
            f"- **{r['factor']}** ({r['relationship']}, Severity: `{r['impact_severity']}`)"
        )
