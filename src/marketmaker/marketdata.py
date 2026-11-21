"""Seeded synthetic market-event generator.

**The dynamics here are deliberately simplified.** A latent "fundamental" price follows
GBM or an Ornstein-Uhlenbeck process. Background liquidity providers (LPs) post limit
orders at geometric distances around the fundamental and cancel them randomly or when
they go stale. Noise traders send market orders (with configurable imbalance); a fraction
of takers are *informed* — they trade in the direction of the fundamental move over the
next few steps — which creates genuine adverse selection for passive quotes.

The generator produces an immutable, time-ordered list of :class:`MarketEvent`. Both
strategies replay the *same* list against independent exchanges. Strategies never see
the fundamental price — only the resulting order book.

Independent RNG streams (via ``SeedSequence.spawn``) are used for the price path, LP
behaviour and taker flow, so changing e.g. LP parameters does not change the price path.

A CSV adapter (:func:`load_price_csv`) can replace the synthetic fundamental with a
local series of mid prices. The order book is still **synthetic** in that case — it is
not historical level-2 data and must not be described as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MarketConfig, RegimeSegment, SimulationConfig
from .models import EventKind, MarketEvent, Side

LP_OWNER = "LP"
TAKER_OWNER = "TAKER"

REGIME_PRESETS: dict[str, dict[str, float]] = {
    "baseline": {},
    "low_liquidity": {"liquidity_multiplier": 0.3, "cancel_multiplier": 2.0},
    "volatility_spike": {
        "volatility_multiplier": 4.0,
        "liquidity_multiplier": 0.7,
        "flow_multiplier": 1.5,
    },
    "trend": {"drift_per_sec": 0.0004, "flow_imbalance": 0.15},
}


@dataclass(frozen=True)
class RegimeSchedule:
    labels: list[str]
    vol_mult: np.ndarray
    liq_mult: np.ndarray
    flow_mult: np.ndarray
    cancel_mult: np.ndarray
    drift: np.ndarray
    imbalance: np.ndarray


@dataclass(frozen=True)
class MarketDataStream:
    events: list[MarketEvent]
    step_times: np.ndarray  # ms, one per step
    fundamental: np.ndarray  # currency, latent (analysis only — never shown to strategies)
    fundamental_ticks: np.ndarray
    regimes: list[str]
    seed: int
    source: str  # "synthetic" or "csv:<path> (synthetic book)"


def regime_schedule(market: MarketConfig, steps: int) -> RegimeSchedule:
    labels = ["baseline"] * steps
    vol = np.ones(steps)
    liq = np.ones(steps)
    flow = np.ones(steps)
    cancel = np.ones(steps)
    drift = np.full(steps, market.drift_per_sec)
    imb = np.full(steps, market.flow_imbalance)
    for seg in market.regimes:
        a = round(seg.start_frac * steps)
        b = round(seg.end_frac * steps)
        params = _segment_params(seg)
        for i in range(a, b):
            labels[i] = seg.regime
        vol[a:b] = params.get("volatility_multiplier", 1.0)
        liq[a:b] = params.get("liquidity_multiplier", 1.0)
        flow[a:b] = params.get("flow_multiplier", 1.0)
        cancel[a:b] = params.get("cancel_multiplier", 1.0)
        drift[a:b] = params.get("drift_per_sec", market.drift_per_sec)
        imb[a:b] = params.get("flow_imbalance", market.flow_imbalance)
    return RegimeSchedule(labels, vol, liq, flow, cancel, drift, imb)


def _segment_params(seg: RegimeSegment) -> dict[str, float]:
    params = dict(REGIME_PRESETS[seg.regime])
    for key in (
        "volatility_multiplier",
        "liquidity_multiplier",
        "flow_multiplier",
        "cancel_multiplier",
        "drift_per_sec",
        "flow_imbalance",
    ):
        val = getattr(seg, key)
        if val is not None:
            params[key] = float(val)
    return params


def simulate_price_path(
    market: MarketConfig, sched: RegimeSchedule, n: int, dt: float, rng: np.random.Generator
) -> np.ndarray:
    """Latent fundamental price for ``n`` steps (n may exceed len(sched) for look-forward
    of informed traders; extra steps reuse the last regime)."""
    s0 = market.initial_price
    z = rng.standard_normal(n)
    path = np.empty(n)
    path[0] = s0
    m = len(sched.vol_mult)
    for i in range(1, n):
        j = min(i, m - 1)
        sigma = market.volatility * sched.vol_mult[j]
        mu = sched.drift[j]
        prev = path[i - 1]
        if market.process == "gbm":
            path[i] = prev * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[i])
        else:  # arithmetic OU around the initial price, drift shifts the anchor
            anchor = s0 * (1.0 + mu * i * dt)
            path[i] = (
                prev + market.ou_kappa * (anchor - prev) * dt + sigma * s0 * np.sqrt(dt) * z[i]
            )
    return np.maximum(path, 1e-9)


def load_price_csv(path: str | Path) -> np.ndarray:
    """Load a local CSV of mid prices. Schema: a ``mid`` column (or ``close``) with finite,
    positive floats; optional ``timestamp`` column must be monotonically increasing."""
    df = pd.read_csv(path)
    col = "mid" if "mid" in df.columns else "close" if "close" in df.columns else None
    if col is None:
        raise ValueError("price CSV must contain a 'mid' or 'close' column")
    values = pd.to_numeric(df[col], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"column {col!r} must contain finite positive numbers")
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if ts.isna().any() or not ts.is_monotonic_increasing:
            raise ValueError("'timestamp' must be parseable and monotonically increasing")
    if len(values) < 2:
        raise ValueError("price CSV needs at least 2 rows")
    return values.to_numpy(dtype=float)


def generate_market_data(
    cfg: SimulationConfig, seed: int | None = None, price_path: np.ndarray | None = None
) -> MarketDataStream:
    seed = cfg.seed if seed is None else seed
    m = cfg.market
    steps = cfg.steps
    dt = cfg.step_ms / 1000.0
    tick = cfg.exchange.tick_size
    horizon = m.informed_horizon_steps
    price_rng, lp_rng, flow_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )
    sched = regime_schedule(m, steps)

    source = "synthetic"
    if price_path is None and m.price_csv:
        price_path = load_price_csv(m.price_csv)
        source = f"csv:{m.price_csv} (mid prices only; order book is synthetic)"
    if price_path is not None:
        if len(price_path) < steps + horizon + 1:
            reps = int(np.ceil((steps + horizon + 1) / len(price_path)))
            price_path = np.concatenate([price_path] * reps)  # repeat to cover the run
        fundamental_full = np.asarray(price_path[: steps + horizon + 1], dtype=float)
        if source == "synthetic":
            source = "external price path (order book is synthetic)"
    else:
        fundamental_full = simulate_price_path(m, sched, steps + horizon + 1, dt, price_rng)

    fund_ticks_full = np.maximum(np.rint(fundamental_full / tick).astype(np.int64), 10)
    events: list[MarketEvent] = []
    outstanding: dict[str, tuple[Side, int]] = {}
    lp_seq = 0
    mkt_seq = 0

    def post(t: int, side: Side, price: int, qty: int) -> None:
        nonlocal lp_seq
        oid = f"lp{lp_seq}"
        lp_seq += 1
        outstanding[oid] = (side, price)
        events.append(MarketEvent(t, EventKind.LP_LIMIT, oid, side, price, qty, LP_OWNER))

    for i in range(steps):
        t = i * cfg.step_ms
        mid = int(fund_ticks_full[i])
        if i == 0:
            for lvl in range(1, m.lp_max_levels + 1):
                post(t, Side.BUY, max(mid - lvl, 1), m.initial_depth_per_level)
                post(t, Side.SELL, mid + lvl, m.initial_depth_per_level)
            continue

        # 1) LP cancellations: stale (through the new mid or too far) or random.
        if outstanding:
            ids = list(outstanding)
            draws = lp_rng.random(len(ids))
            p_cancel = min(1.0, m.lp_cancel_prob * sched.cancel_mult[i])
            far = m.lp_max_levels + 2
            for oid, u in zip(ids, draws, strict=True):
                side, px = outstanding[oid]
                stale = (side is Side.BUY and px >= mid) or (side is Side.SELL and px <= mid)
                if stale or abs(px - mid) > far or u < p_cancel:
                    del outstanding[oid]
                    events.append(MarketEvent(t, EventKind.LP_CANCEL, oid, owner=LP_OWNER))

        # 2) LP arrivals at geometric distance from the fundamental.
        rate = m.lp_arrival_rate * sched.liq_mult[i]
        for side in (Side.BUY, Side.SELL):
            n = int(lp_rng.poisson(rate))
            if n == 0:
                continue
            offsets = np.minimum(lp_rng.geometric(m.lp_level_decay, n), m.lp_max_levels)
            sizes = lp_rng.integers(1, m.lp_max_size + 1, n)
            for off, qty in zip(offsets, sizes, strict=True):
                price = mid - int(off) if side is Side.BUY else mid + int(off)
                if price >= 1:
                    post(t, side, price, int(qty))

        # 3) Taker market orders at random times within the step.
        n_mkt = int(flow_rng.poisson(m.market_order_rate * sched.flow_mult[i]))
        if n_mkt:
            offsets_ms = np.sort(flow_rng.integers(0, cfg.step_ms, n_mkt))
            for off in offsets_ms:
                if flow_rng.random() < m.informed_fraction:
                    future = fundamental_full[i + horizon] - fundamental_full[i]
                    buy = bool(flow_rng.random() < 0.5) if future == 0 else bool(future > 0)
                else:
                    buy = bool(flow_rng.random() < 0.5 + sched.imbalance[i])
                qty = int(flow_rng.integers(1, m.market_order_max_size + 1))
                events.append(
                    MarketEvent(
                        t + int(off),
                        EventKind.NOISE_MARKET,
                        f"mk{mkt_seq}",
                        Side.BUY if buy else Side.SELL,
                        None,
                        qty,
                        TAKER_OWNER,
                    )
                )
                mkt_seq += 1

    events.sort(key=lambda e: (e.timestamp, int(e.kind)))  # stable: keeps generation order
    return MarketDataStream(
        events=events,
        step_times=np.arange(steps, dtype=np.int64) * cfg.step_ms,
        fundamental=fundamental_full[:steps].copy(),
        fundamental_ticks=fund_ticks_full[:steps].copy(),
        regimes=sched.labels,
        seed=seed,
        source=source,
    )
