from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from marketmaker.metrics import (
    MIN_SHARPE_OBS,
    fill_analytics,
    format_value,
    max_drawdown,
    resample_equity,
    safe_ratio,
    sharpe_ratio,
)


def test_max_drawdown_edge_cases() -> None:
    assert max_drawdown([]) == 0.0
    assert max_drawdown([5.0]) == 0.0
    assert max_drawdown([1, 2, 3, 4]) == 0.0
    assert max_drawdown([0, 10, 4, 12, 2, 8]) == 10.0
    assert max_drawdown([-1, -5, -2]) == 4.0


def test_sharpe_insufficient_or_degenerate_returns_none() -> None:
    assert sharpe_ratio([]) is None
    assert sharpe_ratio([1.0] * (MIN_SHARPE_OBS - 1)) is None
    assert sharpe_ratio([1.0] * 100) is None  # zero variance → N/A, never inf
    assert sharpe_ratio([0.0] * 100) is None
    assert sharpe_ratio([np.nan] * 100) is None


def test_sharpe_value() -> None:
    x = np.array([1.0, -1.0, 2.0, 0.0] * 10)
    s = sharpe_ratio(x)
    assert s is not None and math.isfinite(s)
    assert s == pytest.approx(x.mean() / x.std(ddof=1))


def test_safe_ratio_and_format() -> None:
    assert safe_ratio(1, 0) is None
    assert safe_ratio(1, 4) == 0.25
    assert format_value(None) == "N/A"
    assert format_value(True) == "yes"


def test_resample_equity_takes_last_value_at_or_before() -> None:
    ts = pd.DataFrame({"time_ms": [100, 200, 900, 1000, 1500, 2100],
                       "equity": [1, 2, 3, 4, 5, 6]})
    s = resample_equity(ts, 1000)
    assert list(s.index) == [1000, 2000]
    assert list(s.values) == [4, 5]


def test_fill_analytics_signs() -> None:
    ts = pd.DataFrame({"time_ms": [0, 1000, 2000], "mid": [100.0, 100.0, 98.0]})
    fills = pd.DataFrame({
        "time_ms": [500, 1000], "side": ["BUY", "SELL"], "price_ticks": [99, 101],
        "quantity": [1, 1], "mid_at_fill": [100.0, 100.0],
    })
    out = fill_analytics(fills, ts, horizon_ms=1000)
    # buy 1 tick below mid → +1 capture; mid unchanged after 1s → no adverse selection
    assert out.loc[0, "spread_capture_ticks"] == 1.0
    assert out.loc[0, "adverse_selection_ticks"] == 0.0
    # sell at 101 at t=1000; mid at t=2000 is 98 → sell was good, AS negative
    assert out.loc[1, "spread_capture_ticks"] == 1.0
    assert out.loc[1, "markout_ticks"] == 3.0
    assert out.loc[1, "adverse_selection_ticks"] == -2.0


def test_fill_analytics_no_markout_past_end() -> None:
    ts = pd.DataFrame({"time_ms": [0, 1000], "mid": [100.0, 101.0]})
    fills = pd.DataFrame({"time_ms": [900], "side": ["BUY"], "price_ticks": [99],
                          "quantity": [1], "mid_at_fill": [100.0]})
    out = fill_analytics(fills, ts, horizon_ms=1000)
    assert np.isnan(out.loc[0, "markout_ticks"])


def test_fill_analytics_empty() -> None:
    ts = pd.DataFrame({"time_ms": [0], "mid": [100.0]})
    empty = pd.DataFrame(columns=["time_ms", "side", "price_ticks", "quantity", "mid_at_fill"])
    assert fill_analytics(empty, ts, 1000).empty
