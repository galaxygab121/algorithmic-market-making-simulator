from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketmaker.config import SimulationConfig, load_config
from marketmaker.experiments import (
    REGIME_SUITE,
    export_artifacts,
    paired_comparison,
    run_suite,
    with_regime,
)
from marketmaker.marketdata import generate_market_data, load_price_csv
from marketmaker.models import EventKind, MarketEvent, Side
from marketmaker.plots import (
    comparison_figure,
    depth_figure,
    price_and_quotes_figure,
    save_html,
    save_summary_png,
)
from marketmaker.simulation import Simulator, main, run_comparison, run_strategy
from marketmaker.strategies import FixedSpreadStrategy

from .conftest import EXAMPLES


def _stream_from(cfg: SimulationConfig, events: list[MarketEvent]):  # type: ignore[no-untyped-def]
    base = generate_market_data(cfg)
    return replace(base, events=events)


def test_latency_orders_arrive_after_earlier_market_events() -> None:
    cfg = SimulationConfig(steps=10, step_ms=100).with_overrides(**{
        "exchange.order_latency_ms": 20, "strategies.baseline.half_spread_ticks": 2,
        "strategies.baseline.decision_interval_ms": 1000})
    events = [
        MarketEvent(0, EventKind.LP_LIMIT, "lb", Side.BUY, 990, 50, "LP"),
        MarketEvent(0, EventKind.LP_LIMIT, "la", Side.SELL, 1010, 50, "LP"),
        # before the MM quote reaches the exchange (t=20): must trade with LP at 1010
        MarketEvent(10, EventKind.NOISE_MARKET, "m1", Side.BUY, None, 1, "TAKER"),
        # after arrival: must trade with the MM ask at 1002
        MarketEvent(25, EventKind.NOISE_MARKET, "m2", Side.BUY, None, 1, "TAKER"),
    ]
    sim = Simulator(cfg, _stream_from(cfg, events),
                    FixedSpreadStrategy(cfg.strategies.baseline))
    run = sim.run()
    trades = sim.exchange.trades
    assert [(t.timestamp, t.price_ticks, t.maker_owner) for t in trades] == [
        (10, 1010, "LP"), (25, 1002, "MM")]
    acks = run.orders[run.orders["event"] == "ack"]
    assert (acks["time_ms"] == 20).all()
    assert run.fills.iloc[0]["liquidity"] == "MAKER"


def test_message_latencies_respected_in_full_run(small_cfg: SimulationConfig) -> None:
    run = run_strategy(small_cfg, generate_market_data(small_cfg), "inventory_aware")
    log = run.orders
    sub = log[log["event"] == "submit"].set_index("order_id")["time_ms"]
    arrivals = log[log["event"].isin(["ack", "exchange_reject"])].set_index("order_id")[
        "time_ms"]
    joined = arrivals.to_frame("arr").join(sub.to_frame("sub"), how="inner")
    assert len(joined) > 10
    assert (joined["arr"] - joined["sub"] == small_cfg.exchange.order_latency_ms).all()
    creq = log[log["event"] == "cancel_request"].groupby("order_id")["time_ms"].first()
    cack = log[log["event"] == "cancel_ack"].groupby("order_id")["time_ms"].first()
    j = cack.to_frame("ack").join(creq.to_frame("req"), how="inner")
    assert len(j) > 0
    assert (j["ack"] - j["req"] == small_cfg.exchange.cancel_latency_ms).all()


def test_no_lookahead_future_events_do_not_change_past(small_cfg: SimulationConfig) -> None:
    """Dropping every exogenous event after T must leave all decisions before T intact."""
    stream = generate_market_data(small_cfg)
    cut = small_cfg.duration_ms // 2
    truncated = replace(stream, events=[e for e in stream.events if e.timestamp <= cut])
    for name in ("baseline", "inventory_aware"):
        from marketmaker.strategies import build_strategy

        def mk(name: str = name):  # type: ignore[no-untyped-def]
            return build_strategy(name, small_cfg.strategies.baseline,
                                  small_cfg.strategies.inventory_aware)

        full = Simulator(small_cfg, stream, mk()).run()
        part = Simulator(small_cfg, truncated, mk()).run()
        a = full.orders[full.orders["time_ms"] <= cut].reset_index(drop=True)
        b = part.orders[part.orders["time_ms"] <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        qa = full.quotes[full.quotes["time_ms"] <= cut].reset_index(drop=True)
        qb = part.quotes[part.quotes["time_ms"] <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(qa, qb)


@pytest.mark.parametrize("regime", REGIME_SUITE)
def test_both_strategies_run_in_every_regime(small_cfg: SimulationConfig, regime: str) -> None:
    cfg = with_regime(small_cfg, regime).with_overrides(steps=400)
    res = run_comparison(cfg)
    assert set(res.runs) == {"baseline", "inventory_aware"}
    for run in res.runs.values():
        m = run.metrics
        assert abs(m["pnl_identity_residual"]) < 1e-6
        assert run.timeseries["inventory"].abs().max() <= cfg.risk.position_limit
        assert len(run.timeseries) == cfg.steps
        assert m["trade_count"] == len(run.fills)
        # MM inventory is exactly the signed sum of its fills
        signed = np.where(run.fills["side"] == "BUY", 1, -1) * run.fills["quantity"]
        assert int(signed.sum()) == m["final_inventory"]
        # conservation across all market participants
        assert sum(run.all_owner_inventory.values()) == 0


def test_equity_accounting_consistent_through_run(small_cfg: SimulationConfig) -> None:
    res = run_comparison(small_cfg)
    tv = small_cfg.exchange.tick_value
    for run in res.runs.values():
        ts = run.timeseries
        rhs = ts["realized_pnl"] + ts["unrealized_pnl"] - ts["fees"]
        np.testing.assert_allclose(ts["equity"], rhs, atol=1e-6)
        mark = ts["mid"].astype(float)
        np.testing.assert_allclose(ts["equity"], ts["cash"] + ts["inventory"] * mark * tv
                                   - ts["fees"], atol=1e-6)
        assert run.fills["fee"].sum() == pytest.approx(ts["fees"].iloc[-1])


def test_export_artifacts(tmp_path: Path, small_cfg: SimulationConfig) -> None:
    res = run_comparison(small_cfg)
    out = export_artifacts(res, tmp_path / "runs", timestamp="20260101T000000Z")
    assert out.name == f"20260101T000000Z_seed{small_cfg.seed}"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["simulated"] is True and manifest["seed"] == small_cfg.seed
    assert manifest["config"]["steps"] == small_cfg.steps
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"baseline", "inventory_aware"}
    for name in metrics:
        for kind in ("timeseries", "fills", "orders", "quotes", "depth"):
            assert (out / f"{name}_{kind}.csv").exists()
        fills = pd.read_csv(out / f"{name}_fills.csv")
        assert len(fills) == metrics[name]["trade_count"]


def test_charts_render_offline(tmp_path: Path, small_cfg: SimulationConfig) -> None:
    res = run_comparison(small_cfg)
    run = res.runs["inventory_aware"]
    figs = [price_and_quotes_figure(run, small_cfg.exchange.tick_size),
            comparison_figure(res), depth_figure(run, None, small_cfg.exchange.tick_size)]
    for i, fig in enumerate(figs):
        assert len(fig.data) > 0
        p = save_html(fig, tmp_path / f"f{i}.html")
        html = p.read_text(encoding="utf-8")
        # plotly.js is embedded inline: no external <script src="http..."> needed
        assert re.search(r"<script[^>]+src=[\"']https?://", html) is None
        assert len(html) > 1_000_000
    png = save_summary_png(res, tmp_path / "summary.png")
    assert png.stat().st_size > 10_000
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_matches_engine(tmp_path: Path) -> None:
    """CLI and the library entry point (used by the dashboard) agree exactly."""
    cfg_path = EXAMPLES / "baseline.yaml"
    assert main(["--config", str(cfg_path), "--seed", "5", "--steps", "300",
                 "--output", str(tmp_path)]) == 0
    run_dir = next(tmp_path.iterdir())
    cli_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = load_config(cfg_path).with_overrides(steps=300)
    direct = run_comparison(cfg, seed=5)
    for name, run in direct.runs.items():
        for key in ("final_equity_mtm", "trade_count", "volume", "final_inventory"):
            assert cli_metrics[name][key] == pytest.approx(run.metrics[key])


def test_example_configs_load_and_stress_demo_shows_fills_and_kill(
    stress_cfg: SimulationConfig,
) -> None:
    for p in EXAMPLES.glob("*.yaml"):
        load_config(p)
    res = run_comparison(stress_cfg)
    for run in res.runs.values():
        assert run.metrics["trade_count"] > 0
        assert run.metrics["kill_switch_tripped"]


def test_suite_and_paired_comparison(small_cfg: SimulationConfig) -> None:
    suite = run_suite(small_cfg.with_overrides(steps=200), seeds=[1, 2, 3],
                      regimes=("baseline", "trend"))
    assert len(suite) == 2 * 3 * 2
    summary = paired_comparison(suite)
    assert list(summary["regime"]) == ["baseline", "trend"]
    assert (summary["n_seeds"] == 3).all()


def test_csv_price_adapter(tmp_path: Path, small_cfg: SimulationConfig) -> None:
    p = tmp_path / "mids.csv"
    mids = 100 + np.cumsum(np.full(50, 0.01))
    pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=50, freq="s"),
                  "mid": mids}).to_csv(p, index=False)
    np.testing.assert_allclose(load_price_csv(p), mids)
    cfg = small_cfg.with_overrides(**{"market.price_csv": str(p), "steps": 100})
    stream = generate_market_data(cfg)
    assert "synthetic" in stream.source  # never presented as historical L2 data
    assert stream.fundamental[0] == pytest.approx(mids[0])
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"mid": [100, -1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_price_csv(bad)
    with pytest.raises(ValueError):
        pd.DataFrame({"price": [1, 2]}).to_csv(bad, index=False)
        load_price_csv(bad)


def test_config_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        SimulationConfig.model_validate({"steps": 0})
    with pytest.raises(ValueError):
        SimulationConfig.model_validate({"market": {"regimes": [
            {"regime": "trend", "start_frac": 0.5, "end_frac": 0.2}]}})
    with pytest.raises(ValueError):
        SimulationConfig.model_validate({"unknown_key": 1})
    with pytest.raises(KeyError):
        SimulationConfig().with_overrides(**{"market.nope": 1})
