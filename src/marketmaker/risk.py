"""Pre-trade risk checks and the kill switch.

Pre-trade (every new order):
  * kill switch not engaged
  * ``qty <= max_order_qty``
  * worst-case position: current inventory plus *all* open same-side quantity (including
    orders still in flight or pending cancel, since they can still fill) plus this order
    must stay within ``±position_limit``
  * worst-case notional ``|position| * price * tick_value <= max_notional``

Monitoring (every sample): if peak-to-trough equity drawdown exceeds ``max_drawdown``
(or the position limit is somehow breached) the kill switch engages: the agent cancels
all resting quotes immediately (bypassing throttle and minimum quote lifetime) and no new
orders are accepted for the rest of the run. Inventory is *not* force-liquidated; it is
marked to market and closed by the end-of-run liquidation convention, unless
``flatten_on_kill`` is set, in which case the agent also sends one taker market order
to close the position (it may only partially fill in a thin book).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .config import RiskConfig
from .models import Side


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class KillEvent:
    timestamp: int
    reason: str
    equity: float
    inventory: int


class RiskManager:
    def __init__(self, cfg: RiskConfig, tick_value: float) -> None:
        self.cfg = cfg
        self.tick_value = tick_value
        self.killed = False
        self.kill_event: KillEvent | None = None
        self.peak_equity = 0.0
        self.max_drawdown_seen = 0.0
        self.rejections: Counter[str] = Counter()

    def check_order(
        self,
        side: Side,
        price_ticks: int,
        qty: int,
        inventory: int,
        open_buy_qty: int,
        open_sell_qty: int,
    ) -> RiskDecision:
        decision = self._check(side, price_ticks, qty, inventory, open_buy_qty, open_sell_qty)
        if not decision.allowed:
            self.rejections[decision.reason] += 1
        return decision

    def _check(
        self,
        side: Side,
        price_ticks: int,
        qty: int,
        inventory: int,
        open_buy_qty: int,
        open_sell_qty: int,
    ) -> RiskDecision:
        if self.killed:
            return RiskDecision(False, "kill switch engaged")
        if not self.cfg.enabled:
            return RiskDecision(True)
        if qty > self.cfg.max_order_qty:
            return RiskDecision(False, "max order quantity")
        if side is Side.BUY:
            worst = inventory + open_buy_qty + qty
        else:
            worst = inventory - open_sell_qty - qty
        if abs(worst) > self.cfg.position_limit:
            return RiskDecision(False, "position limit")
        if abs(worst) * price_ticks * self.tick_value > self.cfg.max_notional:
            return RiskDecision(False, "max notional")
        return RiskDecision(True)

    def update(self, now: int, equity: float, inventory: int) -> KillEvent | None:
        """Monitor running equity; returns a KillEvent the first time a limit trips."""
        self.peak_equity = max(self.peak_equity, equity)
        dd = self.peak_equity - equity
        self.max_drawdown_seen = max(self.max_drawdown_seen, dd)
        if self.killed or not self.cfg.enabled:
            return None
        if dd > self.cfg.max_drawdown:
            return self.trigger(now, f"drawdown {dd:.2f}>{self.cfg.max_drawdown:g}",
                                equity, inventory)
        if abs(inventory) > self.cfg.position_limit:
            return self.trigger(now, "position limit breached", equity, inventory)
        return None

    def trigger(self, now: int, reason: str, equity: float, inventory: int) -> KillEvent:
        self.killed = True
        self.kill_event = KillEvent(now, reason, equity, inventory)
        return self.kill_event
