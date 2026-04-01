"""Plotly chart generators for the live trading performance dashboard."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_equity_curve(balances: pd.DataFrame) -> Optional[go.Figure]:
    """Line chart of account value over time."""
    if balances.empty or "total_usd" not in balances.columns:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=balances["timestamp"], y=balances["total_usd"],
        mode="lines", name="Account Value",
        line=dict(color="#2196F3", width=2),
    ))

    start_val = float(balances["total_usd"].iloc[0])
    fig.add_hline(
        y=start_val, line_dash="dash", line_color="gray",
        annotation_text=f"Start: ${start_val:,.2f}",
    )

    fig.update_layout(
        title="Account Equity Curve",
        xaxis_title="Date", yaxis_title="USD",
        template="plotly_dark", height=450,
        yaxis=dict(tickformat="$,.2f"),
    )
    return fig


def create_pnl_per_trade(closes: pd.DataFrame) -> Optional[go.Figure]:
    """Bar chart of P&L per closed trade."""
    if closes.empty or "net_pnl" not in closes.columns:
        return None

    colors = ["#4CAF50" if p > 0 else "#F44336" for p in closes["net_pnl"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=closes["close_timestamp"], y=closes["net_pnl"],
        marker_color=colors, name="P&L",
        text=[f"{s}<br>${p:+.2f}" for s, p in zip(closes["symbol"], closes["net_pnl"])],
        hoverinfo="text",
    ))

    fig.update_layout(
        title="P&L Per Trade",
        xaxis_title="Date", yaxis_title="USD",
        template="plotly_dark", height=400,
        yaxis=dict(tickformat="$,.2f"),
    )
    return fig


def create_cumulative_pnl(closes: pd.DataFrame) -> Optional[go.Figure]:
    """Running sum of net P&L over time."""
    if closes.empty or "net_pnl" not in closes.columns:
        return None

    cum_pnl = closes["net_pnl"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=closes["close_timestamp"], y=cum_pnl,
        mode="lines", name="Cumulative P&L",
        line=dict(color="#4CAF50", width=2),
        fill="tozeroy",
        fillcolor="rgba(76, 175, 80, 0.15)",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Cumulative P&L",
        xaxis_title="Date", yaxis_title="USD",
        template="plotly_dark", height=400,
        yaxis=dict(tickformat="$,.2f"),
    )
    return fig


def create_cumulative_fees(closes: pd.DataFrame) -> Optional[go.Figure]:
    """Running sum of fees paid over time."""
    if closes.empty or "fee_total" not in closes.columns:
        return None

    cum_fees = closes["fee_total"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=closes["close_timestamp"], y=cum_fees,
        mode="lines", name="Cumulative Fees",
        line=dict(color="#FF9800", width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 152, 0, 0.15)",
    ))

    fig.update_layout(
        title="Cumulative Fees Paid",
        xaxis_title="Date", yaxis_title="USD",
        template="plotly_dark", height=350,
        yaxis=dict(tickformat="$,.2f"),
    )
    return fig


def create_summary_gauges(stats: dict) -> Optional[go.Figure]:
    """Indicator gauges for key performance metrics."""
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
        subplot_titles=["Win Rate", "Profit Factor", "Total Net P&L", "Total Fees"],
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=stats.get("win_rate", 0),
        number=dict(suffix="%"),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color="#4CAF50"),
            steps=[
                dict(range=[0, 40], color="#FFCDD2"),
                dict(range=[40, 60], color="#FFF9C4"),
                dict(range=[60, 100], color="#C8E6C9"),
            ],
        ),
    ), row=1, col=1)

    pf = stats.get("profit_factor", 0)
    pf_display = min(pf, 10) if pf != float("inf") else 10
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pf_display,
        number=dict(valueformat=".2f"),
        gauge=dict(
            axis=dict(range=[0, 5]),
            bar=dict(color="#2196F3"),
            steps=[
                dict(range=[0, 1], color="#FFCDD2"),
                dict(range=[1, 2], color="#FFF9C4"),
                dict(range=[2, 5], color="#C8E6C9"),
            ],
        ),
    ), row=1, col=2)

    net_pnl = stats.get("total_net_pnl", 0)
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=net_pnl,
        number=dict(prefix="$", valueformat=",.2f"),
        delta=dict(reference=0, valueformat=",.2f"),
    ), row=1, col=3)

    fig.add_trace(go.Indicator(
        mode="number",
        value=stats.get("total_fees", 0),
        number=dict(prefix="$", valueformat=",.2f"),
    ), row=1, col=4)

    fig.update_layout(
        template="plotly_dark", height=300,
        margin=dict(t=60, b=20),
    )
    return fig


def create_pnl_by_symbol(closes: pd.DataFrame) -> Optional[go.Figure]:
    """Horizontal bar chart of net P&L grouped by symbol."""
    if closes.empty or "net_pnl" not in closes.columns:
        return None

    by_symbol = closes.groupby("symbol")["net_pnl"].sum().sort_values()
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in by_symbol]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_symbol.values, y=by_symbol.index,
        orientation="h", marker_color=colors,
    ))

    fig.update_layout(
        title="Net P&L by Symbol",
        xaxis_title="USD", yaxis_title="",
        template="plotly_dark",
        height=max(300, len(by_symbol) * 25 + 100),
        xaxis=dict(tickformat="$,.2f"),
    )
    return fig


def save_chart(fig: go.Figure, path: str) -> None:
    """Save a Plotly figure as HTML (and PNG if kaleido is available)."""
    fig.write_html(path + ".html")
    try:
        fig.write_image(path + ".png")
    except Exception:
        pass  # kaleido not installed — skip static export
