from __future__ import annotations

import math

from marketmaker.config import BaselineStrategyConfig, InventoryStrategyConfig
from marketmaker.strategies import (
    FixedSpreadStrategy,
    InventoryAwareStrategy,
    MarketView,
    VolatilityEstimator,
    build_strategy,
    clamp_post_only,
)


def view(mid: float | None = 1000.0, inv: int = 0, bb: int | None = 990,
         ba: int | None = 1010, sigma: float = 2.0) -> MarketView:
    return MarketView(0, bb, ba, mid, inv, sigma)


def test_baseline_symmetric_quotes() -> None:
    s = FixedSpreadStrategy(BaselineStrategyConfig(half_spread_ticks=3, quote_size=4))
    t = s.target(view())
    assert (t.bid_ticks, t.ask_ticks) == (997, 1003)
    assert t.bid_qty == t.ask_qty == 4


def test_baseline_half_tick_mid_rounds_outward() -> None:
    s = FixedSpreadStrategy(BaselineStrategyConfig(half_spread_ticks=2))
    t = s.target(view(mid=1000.5))
    assert (t.bid_ticks, t.ask_ticks) == (998, 1003)


def test_baseline_inventory_caps_disable_side() -> None:
    cfg = BaselineStrategyConfig(max_inventory=10, quote_size=5)
    s = FixedSpreadStrategy(cfg)
    long_cap = s.target(view(inv=10))
    assert long_cap.bid_ticks is None and long_cap.ask_ticks is not None
    short_cap = s.target(view(inv=-10))
    assert short_cap.ask_ticks is None and short_cap.bid_ticks is not None
    near = s.target(view(inv=8))
    assert near.bid_qty == 2  # never quote beyond the cap


def test_no_quote_without_mid() -> None:
    for s in (FixedSpreadStrategy(BaselineStrategyConfig()),
              InventoryAwareStrategy(InventoryStrategyConfig())):
        t = s.target(view(mid=None))
        assert t.bid_ticks is None and t.ask_ticks is None


def test_reservation_price_skews_against_inventory() -> None:
    s = InventoryAwareStrategy(InventoryStrategyConfig(gamma=0.01, horizon_sec=10))
    r0, h0 = s.reservation_and_half_spread(1000.0, 0, 2.0)
    r_long, h_long = s.reservation_and_half_spread(1000.0, 10, 2.0)
    r_short, _ = s.reservation_and_half_spread(1000.0, -10, 2.0)
    assert r0 == 1000.0
    assert r_long < r0 < r_short
    # r = s - q * gamma * sigma^2 * tau
    assert r_long == 1000.0 - 10 * 0.01 * 4.0 * 10
    assert h0 == h_long  # spread does not depend on inventory


def test_spread_formula_and_bounds() -> None:
    cfg = InventoryStrategyConfig(gamma=0.1, horizon_sec=1.0, k=1.5, max_spread_ticks=20)
    s = InventoryAwareStrategy(cfg)
    _, half = s.reservation_and_half_spread(1000.0, 0, 3.0)
    expected = 0.1 * 9 * 1 + (2 / 0.1) * math.log(1 + 0.1 / 1.5)
    assert half == expected / 2
    _, half_hi = s.reservation_and_half_spread(1000.0, 0, 1000.0)
    assert half_hi == 10.0  # capped at max_spread/2
    _, half_lo = s.reservation_and_half_spread(1000.0, 0, 0.0)
    assert half_lo >= cfg.min_half_spread_ticks


def test_inventory_aware_quotes_min_one_tick_and_never_cross() -> None:
    cfg = InventoryStrategyConfig(gamma=0.01, horizon_sec=10, max_inventory=1000)
    s = InventoryAwareStrategy(cfg)
    for inv in range(-200, 201, 7):
        for sigma in (0.0, 0.5, 2.0, 10.0):
            t = s.target(view(inv=inv, sigma=sigma, bb=995, ba=1005))
            if t.bid_ticks is not None and t.ask_ticks is not None:
                assert t.ask_ticks - t.bid_ticks >= 1
                assert t.ask_ticks - t.bid_ticks <= cfg.max_spread_ticks
            if t.bid_ticks is not None:
                assert t.bid_ticks < 1005
            if t.ask_ticks is not None:
                assert t.ask_ticks > 995


def test_clamp_post_only() -> None:
    assert clamp_post_only(1006, 1010, 995, 1005) == (1004, 1010)
    assert clamp_post_only(990, 994, 995, 1005) == (990, 996)
    assert clamp_post_only(0, 5, None, None) == (None, 5)
    b, a = clamp_post_only(1000, 999, None, None)
    assert b is not None and a is not None and b < a


def test_volatility_estimator_uses_only_past_and_converges() -> None:
    est = VolatilityEstimator(prior_ticks=1.0, halflife_sec=1.0)
    assert est.update(0, 1000.0) == 1.0  # first observation: no change yet
    mid = 1000.0
    for i in range(1, 2000):
        mid += 3.0 if i % 2 else -3.0  # |dmid| = 3 ticks every 1 s → sigma ≈ 3
        est.update(i * 1000, mid)
    assert abs(est.sigma - 3.0) < 1e-6
    assert est.update(2_000_000, None) == est.sigma


def test_build_strategy_names() -> None:
    b, i = BaselineStrategyConfig(), InventoryStrategyConfig()
    assert build_strategy("baseline", b, i).name == "baseline"
    assert build_strategy("inventory_aware", b, i).name == "inventory_aware"
    try:
        build_strategy("nope", b, i)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
