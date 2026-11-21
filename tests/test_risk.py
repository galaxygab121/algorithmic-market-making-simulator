from __future__ import annotations

from marketmaker.config import RiskConfig, SimulationConfig
from marketmaker.models import Side
from marketmaker.risk import RiskManager
from marketmaker.simulation import run_comparison


def rm(**kw: float) -> RiskManager:
    return RiskManager(RiskConfig(**kw), tick_value=1.0)  # type: ignore[arg-type]


def test_max_order_qty() -> None:
    r = rm(max_order_qty=5)
    assert r.check_order(Side.BUY, 100, 5, 0, 0, 0).allowed
    d = r.check_order(Side.BUY, 100, 6, 0, 0, 0)
    assert not d.allowed and d.reason == "max order quantity"


def test_position_limit_counts_open_orders_worst_case() -> None:
    r = rm(position_limit=10, max_notional=1e9)
    assert r.check_order(Side.BUY, 100, 5, inventory=5, open_buy_qty=0, open_sell_qty=0).allowed
    # 5 held + 3 open bids + 5 new = 13 > 10
    d = r.check_order(Side.BUY, 100, 5, inventory=5, open_buy_qty=3, open_sell_qty=0)
    assert not d.allowed and d.reason == "position limit"
    # selling reduces exposure, so it is allowed
    assert r.check_order(Side.SELL, 100, 5, inventory=5, open_buy_qty=3, open_sell_qty=0).allowed
    d = r.check_order(Side.SELL, 100, 6, inventory=-5, open_buy_qty=0, open_sell_qty=0)
    assert not d.allowed


def test_max_notional() -> None:
    r = rm(position_limit=1000, max_notional=1000.0)
    assert r.check_order(Side.BUY, 100, 10, 0, 0, 0).allowed  # 10*100*1 = 1000
    d = r.check_order(Side.BUY, 100, 11, 0, 0, 0)
    assert not d.allowed and d.reason == "max notional"


def test_drawdown_triggers_kill_switch_once() -> None:
    r = rm(max_drawdown=10.0)
    assert r.update(0, 5.0, 0) is None
    assert r.update(1, 20.0, 0) is None  # new peak
    assert r.update(2, 10.5, 0) is None  # dd 9.5
    ev = r.update(3, 9.0, 3)  # dd 11
    assert ev is not None and r.killed and ev.timestamp == 3 and ev.inventory == 3
    assert r.update(4, -100.0, 0) is None  # only reported once
    d = r.check_order(Side.BUY, 100, 1, 0, 0, 0)
    assert not d.allowed and d.reason == "kill switch engaged"
    assert r.rejections["kill switch engaged"] == 1


def test_disabled_risk_allows_everything_but_kill() -> None:
    r = rm(enabled=False, max_order_qty=1)  # type: ignore[arg-type]
    assert r.check_order(Side.BUY, 100, 50, 0, 0, 0).allowed
    assert r.update(0, -1e9, 0) is None


def test_kill_switch_cancels_quotes_in_simulation(stress_cfg: SimulationConfig) -> None:
    """End-to-end: the stress scenario trips the drawdown limit; afterwards the agent has
    no resting quotes and submits nothing new."""
    res = run_comparison(stress_cfg)
    for run in res.runs.values():
        assert run.kill_event is not None
        t_kill = run.kill_event.timestamp
        ts = run.timeseries
        # allow the cancel latency to elapse, then no resting quotes remain
        after = ts[ts["time_ms"] > t_kill + stress_cfg.exchange.cancel_latency_ms + 1]
        assert after["mm_bid"].isna().all() and after["mm_ask"].isna().all()
        submits = run.orders[(run.orders["event"] == "submit")]
        assert (submits["time_ms"] <= t_kill).all()
        assert run.metrics["kill_switch_tripped"]
        # flatten_on_kill is enabled in the demo config
        assert (run.orders["event"] == "flatten_submit").sum() <= 1


def test_position_limit_never_exceeded_in_simulation(small_cfg: SimulationConfig) -> None:
    cfg = small_cfg.with_overrides(**{
        "risk.position_limit": 12, "risk.max_drawdown": 1e9,
        "strategies.baseline.max_inventory": 1000,
        "strategies.inventory_aware.max_inventory": 1000,
        "market.informed_fraction": 0.6,
    })
    res = run_comparison(cfg)
    for run in res.runs.values():
        assert run.timeseries["inventory"].abs().max() <= 12
        assert run.fills["inventory_after"].abs().max() <= 12
        assert run.metrics["risk_rejects"] > 0  # the limit actually bound
