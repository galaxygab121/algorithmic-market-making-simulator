"""Quoting strategies.

A strategy is a pure function of a :class:`MarketView` — the information legitimately
available to the market maker at decision time (the visible book *excluding its own
quotes*, its own inventory, and its own running volatility estimate). It never sees the
latent fundamental price or future events. Order management (latency, minimum quote
lifetime, throttling, risk checks) lives in :mod:`marketmaker.agent` and is shared by
both strategies so the comparison isolates the pricing logic.

1. :class:`FixedSpreadStrategy` — quote ``mid ± half_spread`` with hard inventory caps.
2. :class:`InventoryAwareStrategy` — *inspired by* Avellaneda & Stoikov (2008):

       reservation  r     = mid - q · γ · σ² · τ
       total spread δ     = γ · σ² · τ + (2/γ) · ln(1 + γ/k)
       bid = floor(r - δ/2),   ask = ceil(r + δ/2)

   Deviations from the paper (so this is *not* an exact implementation): τ is a fixed
   rolling risk horizon instead of time-to-terminal ``T - t``; σ is an online EWMA
   estimate of observed mid volatility in ticks/sqrt(s); ``k`` is a configured constant,
   not fitted to fills; prices are rounded outward to integer ticks, half-spread is
   clamped to ``[min_half_spread_ticks, max_spread_ticks/2]``, quotes are clamped so they
   never cross the visible book (they are post-only), and the same hard inventory caps as
   the baseline apply.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import BaselineStrategyConfig, InventoryStrategyConfig, QuotingConfig


@dataclass(frozen=True, slots=True)
class MarketView:
    timestamp: int
    best_bid: int | None  # visible best bid excluding own orders
    best_ask: int | None
    mid: float | None  # current visible mid, or last known if a side is empty
    inventory: int
    sigma_ticks: float  # estimated volatility, ticks per sqrt(second)


@dataclass(frozen=True, slots=True)
class QuoteTarget:
    bid_ticks: int | None
    ask_ticks: int | None
    bid_qty: int
    ask_qty: int
    reservation: float | None = None
    half_spread: float | None = None


NO_QUOTE = QuoteTarget(None, None, 0, 0)


class VolatilityEstimator:
    """EWMA estimate of mid-price volatility in ticks/sqrt(s), using only past mids."""

    def __init__(self, prior_ticks: float, halflife_sec: float) -> None:
        self.var = prior_ticks**2
        self.halflife = halflife_sec
        self._last_t: int | None = None
        self._last_mid: float | None = None

    @property
    def sigma(self) -> float:
        return math.sqrt(self.var)

    def update(self, t_ms: int, mid: float | None) -> float:
        if mid is None:
            return self.sigma
        if self._last_t is not None and self._last_mid is not None and t_ms > self._last_t:
            dt = (t_ms - self._last_t) / 1000.0
            r2 = (mid - self._last_mid) ** 2 / dt
            alpha = 1.0 - 0.5 ** (dt / self.halflife)
            self.var = (1.0 - alpha) * self.var + alpha * r2
        self._last_t, self._last_mid = t_ms, mid
        return self.sigma


def clamp_post_only(
    bid: int | None, ask: int | None, best_bid: int | None, best_ask: int | None
) -> tuple[int | None, int | None]:
    """Keep quotes passive: bid strictly below the visible ask, ask strictly above the
    visible bid, and bid < ask. Returns ``None`` for a side that cannot be quoted."""
    if bid is not None and best_ask is not None:
        bid = min(bid, best_ask - 1)
    if ask is not None and best_bid is not None:
        ask = max(ask, best_bid + 1)
    if bid is not None and bid < 1:
        bid = None
    if bid is not None and ask is not None and bid >= ask:
        ask = bid + 1
    return bid, ask


def apply_inventory_caps(
    target: QuoteTarget, inventory: int, cfg: QuotingConfig
) -> QuoteTarget:
    """Stop quoting the side that would increase |inventory| beyond the soft cap."""
    bid, ask = target.bid_ticks, target.ask_ticks
    bid_qty = min(cfg.quote_size, max(0, cfg.max_inventory - inventory))
    ask_qty = min(cfg.quote_size, max(0, cfg.max_inventory + inventory))
    if bid_qty == 0:
        bid = None
    if ask_qty == 0:
        ask = None
    return QuoteTarget(bid, ask, bid_qty if bid is not None else 0,
                       ask_qty if ask is not None else 0, target.reservation, target.half_spread)


class Strategy(ABC):
    name: str = "strategy"

    def __init__(self, cfg: QuotingConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def target(self, view: MarketView) -> QuoteTarget: ...


class FixedSpreadStrategy(Strategy):
    name = "baseline"

    def __init__(self, cfg: BaselineStrategyConfig) -> None:
        super().__init__(cfg)
        self.half_spread = cfg.half_spread_ticks

    def target(self, view: MarketView) -> QuoteTarget:
        if view.mid is None:
            return NO_QUOTE
        bid = math.floor(view.mid - self.half_spread + 1e-9)
        ask = math.ceil(view.mid + self.half_spread - 1e-9)
        bid_c, ask_c = clamp_post_only(bid, ask, view.best_bid, view.best_ask)
        raw = QuoteTarget(bid_c, ask_c, self.cfg.quote_size, self.cfg.quote_size,
                          view.mid, float(self.half_spread))
        return apply_inventory_caps(raw, view.inventory, self.cfg)


class InventoryAwareStrategy(Strategy):
    name = "inventory_aware"

    def __init__(self, cfg: InventoryStrategyConfig) -> None:
        super().__init__(cfg)
        self.icfg = cfg

    def reservation_and_half_spread(
        self, mid: float, inventory: int, sigma: float
    ) -> tuple[float, float]:
        c = self.icfg
        risk_term = c.gamma * sigma**2 * c.horizon_sec
        reservation = mid - inventory * risk_term
        total = risk_term + (2.0 / c.gamma) * math.log1p(c.gamma / c.k)
        half = min(max(total / 2.0, c.min_half_spread_ticks), c.max_spread_ticks / 2.0)
        return reservation, half

    def target(self, view: MarketView) -> QuoteTarget:
        if view.mid is None:
            return NO_QUOTE
        r, half = self.reservation_and_half_spread(view.mid, view.inventory, view.sigma_ticks)
        bid = math.floor(r - half + 1e-9)
        ask = math.ceil(r + half - 1e-9)
        bid_c, ask_c = clamp_post_only(bid, ask, view.best_bid, view.best_ask)
        width = self.icfg.max_spread_ticks
        if bid_c is not None and ask_c is not None and ask_c - bid_c > width:
            # Enforce the maximum width *after* post-only clamping: keep the side nearest
            # the reservation price (the one the skew wants filled), pull in the other.
            if r >= view.mid:
                ask_c = bid_c + width
            else:
                bid_c = ask_c - width
                if bid_c < 1:
                    bid_c = None
        raw = QuoteTarget(bid_c, ask_c, self.cfg.quote_size, self.cfg.quote_size, r, half)
        return apply_inventory_caps(raw, view.inventory, self.cfg)


STRATEGY_NAMES = ("baseline", "inventory_aware")


def build_strategy(name: str, baseline: BaselineStrategyConfig,
                   inventory: InventoryStrategyConfig) -> Strategy:
    if name == "baseline":
        return FixedSpreadStrategy(baseline)
    if name == "inventory_aware":
        return InventoryAwareStrategy(inventory)
    raise ValueError(f"unknown strategy {name!r}; expected one of {STRATEGY_NAMES}")
