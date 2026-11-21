"""Discrete-event simulation engine and CLI.

Clock and ordering
------------------
All events live in one priority queue keyed by ``(time_ms, EventKind priority, seq)``.
``seq`` is a global insertion counter, so ties are broken deterministically in insertion
order. At equal timestamps the processing order is: agent cancels → agent orders →
LP cancels → LP limit orders → taker market orders → strategy wake-up → accounting sample.
A strategy wake-up at time t therefore sees every exchange event stamped ≤ t and nothing
later (no look-ahead); its messages reach the exchange only after the configured latency.

Usage::

    python -m marketmaker.simulation --config examples/baseline.yaml --seed 42 \
        --output runs/baseline
"""

from __future__ import annotations

import argparse
import heapq
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import DISCLAIMER
from .agent import MarketMakerAgent
from .config import SimulationConfig, load_config
from .exchange import Exchange
from .marketdata import MarketDataStream, generate_market_data
from .metrics import compute_metrics, format_value
from .models import EventKind, Fill, Order, OrderType, Side
from .risk import KillEvent, RiskManager
from .strategies import STRATEGY_NAMES, MarketView, Strategy, VolatilityEstimator, build_strategy

MM_OWNER = "MM"


@dataclass
class RunResult:
    strategy: str
    timeseries: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame
    quotes: pd.DataFrame
    depth: pd.DataFrame
    metrics: dict[str, Any]
    kill_event: KillEvent | None
    events_processed: int
    wall_time_sec: float
    all_owner_inventory: dict[str, int] = field(default_factory=dict)


class Simulator:
    def __init__(self, cfg: SimulationConfig, stream: MarketDataStream, strategy: Strategy,
                 owner: str = MM_OWNER) -> None:
        self.cfg = cfg
        self.stream = stream
        self.strategy = strategy
        self.owner = owner
        self.exchange = Exchange(cfg.exchange)
        self.risk = RiskManager(cfg.risk, cfg.exchange.tick_value)
        self._queue: list[tuple[int, int, int, EventKind, Any]] = []
        self._seq = 0
        self.agent = MarketMakerAgent(
            owner, strategy, strategy.cfg, self.risk, self.exchange,
            cfg.exchange.order_latency_ms, cfg.exchange.cancel_latency_ms, self.schedule,
        )
        prior = getattr(strategy.cfg, "sigma_prior_ticks", None)
        if prior is None:
            prior = cfg.market.volatility * cfg.market.initial_price / cfg.exchange.tick_size
        halflife = getattr(strategy.cfg, "vol_halflife_sec", 30.0)
        self.vol = VolatilityEstimator(max(prior, 1e-6), halflife)
        self.last_mid: float | None = None
        self.now = 0

    def schedule(self, t: int, kind: EventKind, payload: Any) -> None:
        heapq.heappush(self._queue, (t, int(kind), self._seq, kind, payload))
        self._seq += 1

    # --------------------------------------------------------------- views
    def visible_mid(self) -> tuple[int | None, int | None, float | None]:
        book = self.exchange.book
        bb = book.best_excluding(Side.BUY, self.owner)
        ba = book.best_excluding(Side.SELL, self.owner)
        if bb is not None and ba is not None:
            self.last_mid = (bb + ba) / 2.0
        return bb, ba, self.last_mid

    def _view(self, now: int) -> MarketView:
        bb, ba, mid = self.visible_mid()
        sigma = self.vol.update(now, mid)
        inv = self.exchange.ledger(self.owner).inventory
        return MarketView(now, bb, ba, mid, inv, sigma)

    # ----------------------------------------------------------------- run
    def run(self) -> RunResult:
        cfg = self.cfg
        started = time.perf_counter()
        for ev in self.stream.events:
            self.schedule(ev.timestamp, ev.kind, ev)
        interval = self.strategy.cfg.decision_interval_ms
        for t in range(0, cfg.duration_ms, interval):
            self.schedule(t, EventKind.STRATEGY_WAKEUP, None)
        for i in range(1, cfg.steps + 1):
            self.schedule(i * cfg.step_ms, EventKind.SAMPLE, i)

        ledger = self.exchange.ledger(self.owner)
        fill_rows: list[dict[str, Any]] = []
        quote_rows: list[dict[str, Any]] = []
        ts_rows: list[dict[str, Any]] = []
        depth_rows: list[dict[str, Any]] = []

        def on_fill(fill: Fill) -> None:
            self.agent.on_fill(fill)
            fill_rows.append({
                "time_ms": fill.timestamp, "trade_id": fill.trade_id, "order_id": fill.order_id,
                "side": fill.side.value, "price_ticks": fill.price_ticks,
                "quantity": fill.quantity, "liquidity": fill.liquidity.value, "fee": fill.fee,
                "mid_at_fill": self.last_mid, "inventory_after": ledger.inventory,
            })

        self.exchange.subscribe(self.owner, on_fill)
        processed = 0
        end = cfg.duration_ms
        while self._queue:
            t, _prio, _seq, kind, payload = heapq.heappop(self._queue)
            if t > end:
                break
            self.now = t
            processed += 1
            if kind is EventKind.LP_LIMIT or kind is EventKind.NOISE_MARKET:
                ev = payload
                is_mkt = kind is EventKind.NOISE_MARKET
                self.exchange.submit(
                    Order(ev.order_id, ev.side, ev.price_ticks, ev.quantity, ev.owner,
                          order_type=OrderType.MARKET if is_mkt else OrderType.LIMIT), t)
            elif kind is EventKind.LP_CANCEL:
                self.exchange.cancel(payload.order_id, payload.owner, t)
            elif kind is EventKind.AGENT_ORDER:
                self.agent.on_order_arrival(payload, t)
            elif kind is EventKind.AGENT_CANCEL:
                self.agent.on_cancel_arrival(payload, t)
            elif kind is EventKind.STRATEGY_WAKEUP:
                view = self._view(t)
                target = self.agent.on_wakeup(t, view)
                quote_rows.append({
                    "time_ms": t, "mid": view.mid, "sigma_ticks": view.sigma_ticks,
                    "inventory": view.inventory,
                    "target_bid": target.bid_ticks if target else None,
                    "target_ask": target.ask_ticks if target else None,
                    "reservation": target.reservation if target else None,
                    "half_spread": target.half_spread if target else None,
                })
            elif kind is EventKind.SAMPLE:
                ts_rows.append(self._sample(t, int(payload)))
                if payload % cfg.depth_snapshot_every == 0 or payload == cfg.steps:
                    depth_rows.extend(self._depth_snapshot(t))

        kill = self.risk.kill_event
        ts = pd.DataFrame(ts_rows)
        fills = pd.DataFrame(fill_rows, columns=[
            "time_ms", "trade_id", "order_id", "side", "price_ticks", "quantity", "liquidity",
            "fee", "mid_at_fill", "inventory_after"])
        orders = pd.DataFrame(self.agent.log)
        metrics = compute_metrics(
            ts=ts, fills=fills, counts=dict(self.agent.counts),
            submitted_qty=self.agent.submitted_qty, tick_value=cfg.exchange.tick_value,
            taker_fee_bps=cfg.exchange.fees.taker_fee_bps,
            liquidation_penalty_ticks=cfg.liquidation_penalty_ticks,
            sharpe_sample_ms=cfg.sharpe_sample_ms, markout_horizon_ms=cfg.markout_horizon_ms,
            kill_event=kill,
        )
        return RunResult(
            strategy=self.strategy.name, timeseries=ts, fills=fills, orders=orders,
            quotes=pd.DataFrame(quote_rows), depth=pd.DataFrame(depth_rows), metrics=metrics,
            kill_event=kill, events_processed=processed,
            wall_time_sec=time.perf_counter() - started,
            all_owner_inventory={o: led.inventory for o, led in self.exchange.ledgers.items()},
        )

    def _sample(self, t: int, i: int) -> dict[str, Any]:
        ledger = self.exchange.ledger(self.owner)
        bb, ba, mid = self.visible_mid()
        mark = mid if mid is not None else 0.0
        equity = ledger.equity(mark)
        kill = self.risk.update(t, equity, ledger.inventory)
        if kill is not None:
            self.agent.cancel_all(t, f"kill switch: {kill.reason}")
            if self.cfg.risk.flatten_on_kill:
                self.agent.flatten(t)
        step = min(i, self.cfg.steps - 1)
        return {
            "time_ms": t,
            "step": i,
            "regime": self.stream.regimes[step],
            "fundamental": float(self.stream.fundamental[step]),  # analysis only
            "best_bid": bb, "best_ask": ba, "mid": mid,
            "mm_bid": self.agent.resting_price(Side.BUY),
            "mm_ask": self.agent.resting_price(Side.SELL),
            "inventory": ledger.inventory,
            "cash": ledger.cash,
            "fees": ledger.fees,
            "realized_pnl": ledger.position.realized_pnl,
            "unrealized_pnl": ledger.unrealized_pnl(mark),
            "equity": equity,
            "drawdown": self.risk.peak_equity - equity,
            "killed": self.risk.killed,
        }

    def _depth_snapshot(self, t: int, levels: int = 10) -> list[dict[str, Any]]:
        book = self.exchange.book
        rows = []
        for side in (Side.BUY, Side.SELL):
            for price, qty in book.depth(side, levels):
                own = sum(o.remaining_qty for o in book.level(side, price)
                          if o.owner == self.owner)
                rows.append({"time_ms": t, "side": side.value, "price_ticks": price,
                             "quantity": qty, "mm_quantity": own})
        return rows


# ------------------------------------------------------------------- public API
@dataclass
class ComparisonResult:
    config: SimulationConfig
    seed: int
    stream: MarketDataStream
    runs: dict[str, RunResult]

    def metrics_table(self) -> pd.DataFrame:
        return pd.DataFrame({name: r.metrics for name, r in self.runs.items()})


def run_strategy(cfg: SimulationConfig, stream: MarketDataStream, name: str) -> RunResult:
    strategy = build_strategy(name, cfg.strategies.baseline, cfg.strategies.inventory_aware)
    return Simulator(cfg, stream, strategy).run()


def run_comparison(cfg: SimulationConfig, seed: int | None = None,
                   strategies: tuple[str, ...] = STRATEGY_NAMES) -> ComparisonResult:
    """Run each strategy against the *same* seeded market-event stream.
    This is the single entry point used by the CLI, the dashboard and the tests."""
    if seed is not None and seed != cfg.seed:
        cfg = cfg.with_overrides(seed=seed)
    stream = generate_market_data(cfg)
    runs = {name: run_strategy(cfg, stream, name) for name in strategies}
    return ComparisonResult(cfg, cfg.seed, stream, runs)


def _print_summary(result: ComparisonResult) -> None:
    keys = ["final_equity_mtm", "final_equity_liquidated", "realized_pnl", "net_fees",
            "trade_count", "volume", "fill_ratio", "final_inventory", "inventory_std",
            "max_drawdown", "sharpe_per_interval", "avg_quoted_spread_ticks",
            "spread_capture_ticks_per_unit", "adverse_selection_ticks_per_unit",
            "kill_switch_tripped", "kill_switch_reason"]
    names = list(result.runs)
    print(f"\n{DISCLAIMER}")
    print(f"config={result.config.name} seed={result.seed} steps={result.config.steps} "
          f"source={result.stream.source} fingerprint={result.config.fingerprint()}\n")
    print(f"{'metric':<36}" + "".join(f"{n:>22}" for n in names))
    for k in keys:
        print(f"{k:<36}" + "".join(f"{format_value(result.runs[n].metrics[k])[:21]:>22}"
                                   for n in names))
    for n, r in result.runs.items():
        print(f"\n[{n}] {r.events_processed} events in {r.wall_time_sec:.2f}s wall time")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmsim", description="Offline simulated market-making run (paper trading only).")
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument("--steps", type=int, default=None, help="override number of steps")
    p.add_argument("--strategy", choices=[*STRATEGY_NAMES, "both"], default="both")
    p.add_argument("--output", default=None, help="directory for run artifacts")
    return p


def main(argv: list[str] | None = None) -> int:
    from .experiments import export_artifacts

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg = cfg.with_overrides(steps=args.steps)
    names = STRATEGY_NAMES if args.strategy == "both" else (args.strategy,)
    result = run_comparison(cfg, seed=args.seed, strategies=names)
    _print_summary(result)
    if args.output:
        path = export_artifacts(result, args.output)
        print(f"\nArtifacts written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
