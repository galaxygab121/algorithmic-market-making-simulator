"""Performance metrics computed from an actual simulation run.

Conventions (also in the README glossary):

* **Mark-to-market**: open inventory is marked at the visible mid *excluding the market
  maker's own quotes*; if one side of the book is empty the last valid mid is used.
* **Realized P&L**: average-cost method, excluding fees.
* **Unrealized P&L**: ``inventory * (mark - avg_cost) * tick_value``.
* **Equity** = realized + unrealized - net fees = cash + inventory*mark*tick_value - fees.
* **Liquidated equity**: final inventory is assumed closed at
  ``mid ∓ liquidation_penalty_ticks`` and pays the taker fee.
* **Sharpe**: mean / std of equity *changes* over fixed ``sharpe_sample_ms`` intervals,
  risk-free rate 0, **not annualised** (a simulated session has no meaningful year).
  Returns ``None`` (shown as N/A) with fewer than ``MIN_SHARPE_OBS`` observations or zero
  variance — never ±inf.
* **Spread capture** per fill: ``side_sign * (mid_at_fill - fill_price)`` ticks/unit.
* **Markout(h)**: ``side_sign * (mid_{t+h} - fill_price)``.
* **Adverse selection(h)** = spread capture - markout = ``side_sign * (mid_t - mid_{t+h})``;
  positive means the mid moved against the fill.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

MIN_SHARPE_OBS = 30


def max_drawdown(equity: np.ndarray | pd.Series | list[float]) -> float:
    arr = np.asarray(equity, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    return float(np.max(peak - arr))


def sharpe_ratio(pnl_changes: np.ndarray | pd.Series | list[float],
                 min_obs: int = MIN_SHARPE_OBS) -> float | None:
    x = np.asarray(pnl_changes, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < min_obs:
        return None
    sd = float(np.std(x, ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return None
    return float(np.mean(x) / sd)


def safe_ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def resample_equity(ts: pd.DataFrame, sample_ms: int) -> pd.Series:
    """Equity at the last sample at or before each multiple of ``sample_ms``."""
    if ts.empty:
        return pd.Series(dtype=float)
    grid = np.arange(sample_ms, int(ts["time_ms"].iloc[-1]) + 1, sample_ms)
    idx = np.searchsorted(ts["time_ms"].to_numpy(), grid, side="right") - 1
    idx = idx[idx >= 0]
    return pd.Series(ts["equity"].to_numpy()[idx], index=grid[: len(idx)])


def mid_at(ts: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """Observed mid at or before each time (NaN if before the first sample)."""
    t = ts["time_ms"].to_numpy()
    mids = ts["mid"].to_numpy(dtype=float)
    idx = np.searchsorted(t, times, side="right") - 1
    out = np.full(len(times), np.nan)
    ok = idx >= 0
    out[ok] = mids[idx[ok]]
    return out


def fill_analytics(fills: pd.DataFrame, ts: pd.DataFrame, horizon_ms: int) -> pd.DataFrame:
    if fills.empty:
        return fills.assign(spread_capture_ticks=[], markout_ticks=[],
                            adverse_selection_ticks=[])
    sign = np.where(fills["side"] == "BUY", 1.0, -1.0)
    px = fills["price_ticks"].to_numpy(dtype=float)
    mid0 = fills["mid_at_fill"].to_numpy(dtype=float)
    end = int(ts["time_ms"].iloc[-1])
    t_h = fills["time_ms"].to_numpy() + horizon_ms
    mid_h = mid_at(ts, np.minimum(t_h, end))
    mid_h[t_h > end] = np.nan  # no markout without a full horizon
    out = fills.copy()
    out["spread_capture_ticks"] = sign * (mid0 - px)
    out["markout_ticks"] = sign * (mid_h - px)
    out["adverse_selection_ticks"] = sign * (mid0 - mid_h)
    return out


def _wavg(values: pd.Series, weights: pd.Series) -> float | None:
    mask = values.notna()
    w = weights[mask]
    if w.sum() == 0:
        return None
    return float(np.average(values[mask], weights=w))


def compute_metrics(
    *,
    ts: pd.DataFrame,
    fills: pd.DataFrame,
    counts: dict[str, int],
    submitted_qty: int,
    tick_value: float,
    taker_fee_bps: float,
    liquidation_penalty_ticks: float,
    sharpe_sample_ms: int,
    markout_horizon_ms: int,
    kill_event: Any = None,
) -> dict[str, Any]:
    final = ts.iloc[-1]
    inv = int(final["inventory"])
    mark = float(final["mid"]) if pd.notna(final["mid"]) else 0.0
    liq_px = mark - math.copysign(liquidation_penalty_ticks, inv) if inv else mark
    liq_cost = (abs(inv) * liquidation_penalty_ticks * tick_value
                + abs(inv) * liq_px * tick_value * taker_fee_bps / 1e4) if inv else 0.0
    equity = float(final["equity"])

    eq = resample_equity(ts, sharpe_sample_ms)
    changes = eq.diff().dropna().to_numpy()
    fa = fill_analytics(fills, ts, markout_horizon_ms)
    filled_qty = int(fills["quantity"].sum()) if not fills.empty else 0
    both = ts.dropna(subset=["mm_bid", "mm_ask"])
    rebates = float(-fills.loc[fills["fee"] < 0, "fee"].sum()) if not fills.empty else 0.0
    fees_paid = float(fills.loc[fills["fee"] > 0, "fee"].sum()) if not fills.empty else 0.0
    identity_residual = (float(final["realized_pnl"]) + float(final["unrealized_pnl"])
                         - float(final["fees"]) - equity)
    qty = fa["quantity"] if not fa.empty else pd.Series(dtype=float)

    return {
        "final_equity_mtm": equity,
        "final_equity_liquidated": equity - liq_cost,
        "liquidation_cost": liq_cost,
        "realized_pnl": float(final["realized_pnl"]),
        "unrealized_pnl": float(final["unrealized_pnl"]),
        "net_fees": float(final["fees"]),
        "fees_paid": fees_paid,
        "rebates_earned": rebates,
        "pnl_identity_residual": identity_residual,
        "final_inventory": inv,
        "max_abs_inventory": int(ts["inventory"].abs().max()),
        "inventory_std": float(ts["inventory"].std(ddof=0)),
        "max_drawdown": max_drawdown(ts["equity"].to_numpy()),
        "pnl_volatility_per_interval": (float(np.std(changes, ddof=1))
                                        if changes.size >= 2 else None),
        "sharpe_per_interval": sharpe_ratio(changes),
        "sharpe_assumptions": f"{sharpe_sample_ms} ms equity-change samples, rf=0, "
                              f"not annualised, N/A if < {MIN_SHARPE_OBS} obs",
        "trade_count": len(fills),
        "maker_fills": int(counts.get("maker_fills", 0)),
        "taker_fills": int(counts.get("taker_fills", 0)),
        "volume": filled_qty,
        "orders_submitted": int(counts.get("orders_submitted", 0)),
        "cancels_sent": int(counts.get("cancels_sent", 0)),
        "cancels_too_late": int(counts.get("cancels_too_late", 0)),
        "exchange_rejects": int(counts.get("exchange_rejects", 0)),
        "risk_rejects": int(counts.get("risk_rejects", 0)),
        "throttled": int(counts.get("throttled", 0)),
        "fill_ratio": safe_ratio(filled_qty, submitted_qty),
        "avg_quoted_spread_ticks": (float((both["mm_ask"] - both["mm_bid"]).mean())
                                    if not both.empty else None),
        "quote_uptime_two_sided": float(len(both) / len(ts)) if len(ts) else None,
        "spread_capture_ticks_per_unit": (_wavg(fa["spread_capture_ticks"], qty)
                                          if not fa.empty else None),
        "spread_capture_total": (float((fa["spread_capture_ticks"] * qty).sum() * tick_value)
                                 if not fa.empty else 0.0),
        "markout_ticks_per_unit": _wavg(fa["markout_ticks"], qty) if not fa.empty else None,
        "adverse_selection_ticks_per_unit": (_wavg(fa["adverse_selection_ticks"], qty)
                                             if not fa.empty else None),
        "markout_horizon_ms": markout_horizon_ms,
        "kill_switch_tripped": kill_event is not None,
        "kill_switch_time_ms": kill_event.timestamp if kill_event is not None else None,
        "kill_switch_reason": kill_event.reason if kill_event is not None else None,
    }


def format_value(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
    return str(v)
