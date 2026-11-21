"""Chart builders (Plotly for the dashboard, matplotlib for static PNGs).

Everything renders offline: Plotly HTML exports embed plotly.js (``include_plotlyjs=True``)
and matplotlib uses the non-interactive Agg backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import DISCLAIMER
from .simulation import ComparisonResult, RunResult

COLORS = {"baseline": "#2563eb", "inventory_aware": "#d97706"}
SIM_TAG = "SIMULATED"


def _sec(ts: pd.DataFrame) -> pd.Series:
    return ts["time_ms"] / 1000.0


def price_and_quotes_figure(run: RunResult, tick_size: float) -> go.Figure:
    ts = run.timeseries
    fig = go.Figure()
    fig.add_scatter(x=_sec(ts), y=ts["mid"] * tick_size, name="visible mid",
                    line={"color": "#111827", "width": 1.2})
    fig.add_scatter(x=_sec(ts), y=ts["mm_bid"] * tick_size, name="MM bid", mode="lines",
                    line={"color": "#16a34a", "width": 1, "shape": "hv"})
    fig.add_scatter(x=_sec(ts), y=ts["mm_ask"] * tick_size, name="MM ask", mode="lines",
                    line={"color": "#dc2626", "width": 1, "shape": "hv"})
    f = run.fills
    if not f.empty:
        for side, color, sym in (("BUY", "#16a34a", "triangle-up"),
                                 ("SELL", "#dc2626", "triangle-down")):
            s = f[f["side"] == side]
            fig.add_scatter(x=s["time_ms"] / 1000.0, y=s["price_ticks"] * tick_size,
                            mode="markers", name=f"MM {side.lower()} fills",
                            marker={"color": color, "symbol": sym, "size": 6, "opacity": 0.7})
    if run.kill_event is not None:
        fig.add_vline(x=run.kill_event.timestamp / 1000.0, line_dash="dash",
                      line_color="#7c3aed", annotation_text="kill switch")
    fig.update_layout(title=f"{run.strategy}: mid, quotes and fills ({SIM_TAG})",
                      xaxis_title="simulated time (s)", yaxis_title="price",
                      height=440, legend={"orientation": "h", "y": 1.08, "x": 0})
    return fig


def comparison_figure(result: ComparisonResult) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Equity (mark-to-market)", "Inventory", "Drawdown"))
    for name, run in result.runs.items():
        ts = run.timeseries
        c = COLORS.get(name)
        fig.add_scatter(x=_sec(ts), y=ts["equity"], name=name, legendgroup=name,
                        line={"color": c}, row=1, col=1)
        fig.add_scatter(x=_sec(ts), y=ts["inventory"], name=name, legendgroup=name,
                        showlegend=False, line={"color": c, "shape": "hv"}, row=2, col=1)
        fig.add_scatter(x=_sec(ts), y=-ts["drawdown"], name=name, legendgroup=name,
                        showlegend=False, line={"color": c}, row=3, col=1)
        if run.kill_event is not None:
            fig.add_vline(x=run.kill_event.timestamp / 1000.0, line_dash="dot",
                          line_color=c)
    fig.update_layout(height=720, title=f"Baseline vs inventory-aware ({SIM_TAG})")
    fig.update_xaxes(title_text="simulated time (s)", row=3, col=1)
    return fig


def depth_figure(run: RunResult, time_ms: int | None, tick_size: float) -> go.Figure:
    d = run.depth
    fig = go.Figure()
    if d.empty:
        return fig
    times = sorted(d["time_ms"].unique())
    t = times[-1] if time_ms is None else min(times, key=lambda x: abs(x - time_ms))
    snap = d[d["time_ms"] == t]
    for side, color in (("BUY", "#16a34a"), ("SELL", "#dc2626")):
        s = snap[snap["side"] == side]
        fig.add_bar(x=s["price_ticks"] * tick_size, y=s["quantity"] - s["mm_quantity"],
                    name=f"{side.lower()} (others)", marker_color=color, opacity=0.55)
        fig.add_bar(x=s["price_ticks"] * tick_size, y=s["mm_quantity"],
                    name=f"{side.lower()} (MM)", marker_color=color)
    fig.update_layout(barmode="stack", height=360, xaxis_title="price",
                      yaxis_title="resting quantity",
                      title=f"Book depth at t={t / 1000:.1f}s ({SIM_TAG}, synthetic book)")
    return fig


def save_html(fig: go.Figure, path: str | Path) -> Path:
    """Write a fully self-contained HTML file (plotly.js embedded, no CDN)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(p, include_plotlyjs=True, full_html=True)
    return p


def save_summary_png(result: ComparisonResult, path: str | Path) -> Path:
    """Static 4-panel matplotlib summary used for README images."""
    tick = result.config.exchange.tick_size
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    first = next(iter(result.runs.values())).timeseries
    axes[0].plot(_sec(first), first["mid"] * tick, color="#111827", lw=1, label="visible mid")
    for name, run in result.runs.items():
        ts = run.timeseries
        c = COLORS.get(name)
        axes[0].step(_sec(ts), ts["mm_bid"] * tick, where="post", color=c, lw=0.6, alpha=0.6,
                     label=f"{name} bid/ask")
        axes[0].step(_sec(ts), ts["mm_ask"] * tick, where="post", color=c, lw=0.6, alpha=0.6)
        axes[1].plot(_sec(ts), ts["equity"], color=c, label=name)
        axes[2].step(_sec(ts), ts["inventory"], where="post", color=c, label=name)
        axes[3].plot(_sec(ts), -ts["drawdown"], color=c, label=name)
        if run.kill_event is not None:
            for ax in axes:
                ax.axvline(run.kill_event.timestamp / 1000.0, color=c, ls="--", lw=1)
    titles = ["Visible mid and MM quotes", "Equity (mark-to-market, currency)",
              "Inventory (units)", "Drawdown from peak"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("simulated time (s)")
    fig.suptitle(f"{result.config.name} — seed {result.seed} — {DISCLAIMER[:60]}…",
                 fontsize=9)
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p
