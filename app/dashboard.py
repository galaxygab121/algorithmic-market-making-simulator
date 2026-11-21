"""Streamlit dashboard for the offline market-making simulator.

Run:  uv run streamlit run app/dashboard.py

The dashboard calls ``marketmaker.simulation.run_comparison`` — exactly the same engine
as the CLI — so the same config + seed gives identical results in both places. Download
the effective config YAML to reproduce any dashboard run from the command line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:  # normal case: package installed by `uv sync`
    import marketmaker  # noqa: F401
except ModuleNotFoundError:  # e.g. venv .pth ignored — fall back to the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketmaker import DISCLAIMER
from marketmaker.config import SimulationConfig, dump_config_yaml, load_config
from marketmaker.metrics import format_value
from marketmaker.plots import comparison_figure, depth_figure, price_and_quotes_figure
from marketmaker.simulation import ComparisonResult, run_comparison

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

st.set_page_config(page_title="Market-Making Simulator (simulated)", layout="wide")
st.title("Market-making simulator")
st.warning(f"**{DISCLAIMER}** No connection to any exchange or broker; prices and order "
           "flow are synthetic.", icon="⚠️")

presets = sorted(p.stem for p in EXAMPLES.glob("*.yaml"))
with st.sidebar:
    st.header("Scenario")
    preset = st.selectbox("Preset config", presets,
                          index=presets.index("baseline") if "baseline" in presets else 0)
    base = load_config(EXAMPLES / f"{preset}.yaml")
    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=base.seed, step=1)
    steps = st.slider("Simulation length (steps of "
                      f"{base.step_ms} ms)", 200, 10_000, base.steps, step=100)
    st.header("Market")
    vol = st.number_input("Volatility (relative, per √s)", 0.0, 0.01,
                          float(base.market.volatility), step=0.00005, format="%.5f")
    informed = st.slider("Informed taker fraction", 0.0, 1.0,
                         float(base.market.informed_fraction), 0.05)
    st.header("Strategies")
    half_spread = st.slider("Baseline half-spread (ticks)", 1, 20,
                            base.strategies.baseline.half_spread_ticks)
    gamma = st.number_input("Inventory-aware risk aversion γ", 0.0001, 1.0,
                            float(base.strategies.inventory_aware.gamma), step=0.001,
                            format="%.4f")
    inv_limit = st.slider("Strategy inventory cap (units)", 1, 500,
                          base.strategies.baseline.max_inventory)
    st.header("Fees & risk")
    maker_bps = st.number_input("Maker fee (bps, negative = rebate)", -5.0, 5.0,
                                float(base.exchange.fees.maker_fee_bps), step=0.1)
    taker_bps = st.number_input("Taker fee (bps)", 0.0, 10.0,
                                float(base.exchange.fees.taker_fee_bps), step=0.1)
    max_dd = st.number_input("Max drawdown (kill switch)", 1.0, 1e6,
                             float(base.risk.max_drawdown), step=5.0)
    run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)


def effective_config() -> SimulationConfig:
    return base.with_overrides(**{
        "seed": int(seed), "steps": int(steps),
        "market.volatility": float(vol), "market.informed_fraction": float(informed),
        "strategies.baseline.half_spread_ticks": int(half_spread),
        "strategies.baseline.max_inventory": int(inv_limit),
        "strategies.inventory_aware.max_inventory": int(inv_limit),
        "strategies.inventory_aware.gamma": float(gamma),
        "exchange.fees.maker_fee_bps": float(maker_bps),
        "exchange.fees.taker_fee_bps": float(taker_bps),
        "risk.max_drawdown": float(max_dd),
    })


@st.cache_data(show_spinner="Running simulation…")
def cached_run(config_json: str) -> ComparisonResult:
    return run_comparison(SimulationConfig.model_validate_json(config_json))


cfg = effective_config()
if run_clicked or "result_cfg" not in st.session_state:
    st.session_state["result_cfg"] = cfg.model_dump_json()
result = cached_run(st.session_state["result_cfg"])
rcfg = result.config
tick = rcfg.exchange.tick_size

if rcfg.model_dump_json() != cfg.model_dump_json():
    st.info("Controls changed — press **Run Simulation** to update the results.")

st.caption(f"Config **{rcfg.name}** · seed **{result.seed}** · {rcfg.steps} steps × "
           f"{rcfg.step_ms} ms · data: {result.stream.source} · fingerprint "
           f"`{rcfg.fingerprint()}`")
st.download_button("Download effective config (YAML) to reproduce via CLI",
                   dump_config_yaml(rcfg), file_name=f"{rcfg.name}_seed{result.seed}.yaml")

st.subheader("Side-by-side metrics (simulated)")
keys = ["final_equity_mtm", "final_equity_liquidated", "realized_pnl", "net_fees",
        "trade_count", "volume", "fill_ratio", "maker_fills", "taker_fills",
        "final_inventory", "inventory_std", "max_abs_inventory", "max_drawdown",
        "pnl_volatility_per_interval", "sharpe_per_interval", "avg_quoted_spread_ticks",
        "spread_capture_ticks_per_unit", "adverse_selection_ticks_per_unit",
        "orders_submitted", "cancels_sent", "risk_rejects", "kill_switch_tripped",
        "kill_switch_reason"]
table = pd.DataFrame({name: [format_value(r.metrics[k]) for k in keys]
                      for name, r in result.runs.items()}, index=keys)
c1, c2 = st.columns([2, 1])
c1.dataframe(table, use_container_width=True, height=600)
with c2:
    for name, run in result.runs.items():
        m = run.metrics
        st.metric(f"{name} — liquidated equity", format_value(m["final_equity_liquidated"]),
                  help="Simulated; final inventory closed at mid ∓ penalty plus taker fee")
        if run.kill_event is not None:
            st.error(f"{name}: kill switch at t={run.kill_event.timestamp / 1000:.1f}s "
                     f"({run.kill_event.reason})")
    st.caption(next(iter(result.runs.values())).metrics["sharpe_assumptions"])

st.plotly_chart(comparison_figure(result), use_container_width=True)

st.subheader("Quotes, fills and book depth")
name = st.radio("Strategy", list(result.runs), horizontal=True)
run = result.runs[name]
st.plotly_chart(price_and_quotes_figure(run, tick), use_container_width=True)
times = sorted(run.depth["time_ms"].unique()) if not run.depth.empty else [0]
t_sel = st.select_slider("Depth snapshot time (s)", options=times, value=times[-1],
                         format_func=lambda t: f"{t / 1000:.1f}")
st.plotly_chart(depth_figure(run, int(t_sel), tick), use_container_width=True)

with st.expander("Fills"):
    st.dataframe(run.fills, use_container_width=True)
with st.expander("Order / message log"):
    st.dataframe(run.orders, use_container_width=True)
st.caption(DISCLAIMER)
