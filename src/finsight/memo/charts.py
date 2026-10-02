"""Plotly Chart Generators for Institutional Equity Research Reports and Web UI."""

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHART_THEME = {
    "paper_bgcolor": "#0b0f19",
    "plot_bgcolor": "#111827",
    "font_color": "#f9fafb",
    "font_family": "Inter, sans-serif",
    "grid_color": "#1f2937",
    "primary_blue": "#2563eb",
    "accent_teal": "#0ea5e9",
    "bullish_green": "#10b981",
    "bearish_red": "#ef4444",
    "warning_amber": "#f59e0b",
}


def create_price_and_fan_chart(
    history_df: pd.DataFrame,
    fan_points: list[dict[str, Any]],
    ticker: str = "TATAMOTORS.NS",
) -> dict[str, Any]:
    """Builds interactive historical price chart with LSTM quantile prediction cone."""
    df = history_df.tail(120).copy()

    fig = go.Figure()

    # Historical Close
    fig.add_trace(
        go.Scatter(
            x=df["Date"].astype(str).tolist(),
            y=df["Close"].astype(float).tolist(),
            mode="lines",
            name="Historical Price",
            line=dict(color=CHART_THEME["primary_blue"], width=2.5),
        )
    )

    # 50-day and 200-day Moving Averages if available
    if "sma_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"].astype(str).tolist(),
                y=df["sma_50"].astype(float).tolist(),
                mode="lines",
                name="50d SMA",
                line=dict(color="#9ca3af", width=1.2, dash="dash"),
            )
        )

    # Forecast Fan Points
    if fan_points:
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        fan_dates = [
            (last_date + pd.Timedelta(days=int(p["day"] * 1.4))).strftime("%Y-%m-%d")
            for p in fan_points
        ]
        p10 = [float(p["price_p10"]) for p in fan_points]
        p50 = [float(p["price_p50"]) for p in fan_points]
        p90 = [float(p["price_p90"]) for p in fan_points]

        # Connect fan to last price
        last_price = float(df["Close"].iloc[-1])
        conn_dates = [str(df["Date"].iloc[-1])] + fan_dates
        conn_p10 = [last_price] + p10
        conn_p50 = [last_price] + p50
        conn_p90 = [last_price] + p90

        # p90 upper band
        fig.add_trace(
            go.Scatter(
                x=conn_dates,
                y=conn_p90,
                mode="lines",
                name="Optimistic (p90)",
                line=dict(color="rgba(16, 185, 129, 0.0)", width=0),
                showlegend=False,
            )
        )
        # p10 lower band with fill to p90
        fig.add_trace(
            go.Scatter(
                x=conn_dates,
                y=conn_p10,
                mode="lines",
                name="80% Prediction Cone",
                line=dict(color="rgba(16, 185, 129, 0.0)", width=0),
                fill="tonexty",
                fillcolor="rgba(37, 99, 235, 0.18)",
            )
        )
        # p50 median trajectory
        fig.add_trace(
            go.Scatter(
                x=conn_dates,
                y=conn_p50,
                mode="lines",
                name="Base Forecast (p50)",
                line=dict(color=CHART_THEME["bullish_green"], width=2.5, dash="dot"),
            )
        )

    fig.update_layout(
        title=f"<b>{ticker}</b> Price Trajectory & 12M Probabilistic Forecast Fan",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(family=CHART_THEME["font_family"], color=CHART_THEME["font_color"], size=12),
        xaxis=dict(gridcolor=CHART_THEME["grid_color"], showgrid=True),
        yaxis=dict(gridcolor=CHART_THEME["grid_color"], showgrid=True, title="Price (₹)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
    )
    return json.loads(fig.to_json())


def create_backtest_equity_chart(equity_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Generates strategy equity curve vs Buy-and-Hold benchmark with drawdown subplot."""
    if not equity_records:
        return {}

    df = pd.DataFrame(equity_records)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08
    )

    # Equity curves
    fig.add_trace(
        go.Scatter(
            x=df["Date"].tolist(),
            y=df["Strategy_Equity"].astype(float).tolist(),
            mode="lines",
            name="FinSight Model Strategy",
            line=dict(color=CHART_THEME["bullish_green"], width=2.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"].tolist(),
            y=df["Benchmark_Equity"].astype(float).tolist(),
            mode="lines",
            name="Buy & Hold Asset",
            line=dict(color="#6b7280", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Drawdown subplot
    fig.add_trace(
        go.Scatter(
            x=df["Date"].tolist(),
            y=(df["Strategy_Drawdown"] * 100).astype(float).tolist(),
            mode="lines",
            name="Strategy Drawdown",
            line=dict(color=CHART_THEME["bearish_red"], width=1.2),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="<b>Systematic Strategy Backtest (Out-of-Sample)</b> vs Buy & Hold",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(family=CHART_THEME["font_family"], color=CHART_THEME["font_color"], size=12),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_yaxes(
        title_text="Portfolio Value (₹)", row=1, col=1, gridcolor=CHART_THEME["grid_color"]
    )
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1, gridcolor=CHART_THEME["grid_color"])
    fig.update_xaxes(gridcolor=CHART_THEME["grid_color"])
    return json.loads(fig.to_json())


def create_shap_driver_chart(shap_drivers: list[dict[str, Any]]) -> dict[str, Any]:
    """Generates horizontal waterfall/bar chart of SHAP local drivers."""
    if not shap_drivers:
        return {}

    features = [d["feature"] for d in reversed(shap_drivers)]
    impacts = [float(d["shap_contribution"]) for d in reversed(shap_drivers)]
    colors = [
        CHART_THEME["bullish_green"] if imp > 0 else CHART_THEME["bearish_red"] for imp in impacts
    ]

    fig = go.Figure(
        go.Bar(
            x=impacts,
            y=features,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{imp:+.3f}" for imp in impacts],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="<b>Local SHAP Top Feature Drivers (Model Explainability)</b>",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(family=CHART_THEME["font_family"], color=CHART_THEME["font_color"], size=12),
        xaxis=dict(title="SHAP Impact on Bullish Probability", gridcolor=CHART_THEME["grid_color"]),
        yaxis=dict(gridcolor=CHART_THEME["grid_color"]),
        margin=dict(l=120, r=60, t=50, b=40),
    )
    return json.loads(fig.to_json())


def create_peer_comparison_chart(peers: list[dict[str, Any]]) -> dict[str, Any]:
    """Builds peer valuation (P/E vs ROE) scatter comparison."""
    if not peers:
        return {}

    names = [p.get("name", p.get("ticker")) for p in peers]
    pe = [float(p.get("pe", 15)) for p in peers]
    colors = [
        CHART_THEME["primary_blue"] if "Tata" in n else CHART_THEME["accent_teal"] for n in names
    ]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=pe,
            name="P/E Ratio (x)",
            marker=dict(color=colors),
            text=[f"{val:.1f}x" for val in pe],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="<b>Valuation Multiples vs Industry Peers (P/E Ratio)</b>",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(family=CHART_THEME["font_family"], color=CHART_THEME["font_color"], size=12),
        yaxis=dict(title="Trailing P/E Ratio", gridcolor=CHART_THEME["grid_color"]),
        xaxis=dict(gridcolor=CHART_THEME["grid_color"]),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return json.loads(fig.to_json())
