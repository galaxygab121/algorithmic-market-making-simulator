"""Market-maker agent: turns a strategy's quote targets into exchange messages.

Shared by every strategy so comparisons isolate pricing logic. Responsibilities:

* one working quote per side; requote when the target moves by at least
  ``requote_threshold_ticks`` (or a side is disabled)
* respect ``min_quote_lifetime_ms`` before cancel/replace of a quote
* message throttle: at most ``max_messages_per_sec`` new orders + cancels in any rolling
  1-second window (risk-driven cancels bypass the throttle and lifetime, but are counted)
* inbound latency: new orders arrive ``order_latency_ms`` later, cancels arrive
  ``cancel_latency_ms`` later — a quote can still be filled while its cancel is in flight
* pre-trade risk check on every new order (worst-case exposure includes in-flight and
  pending-cancel orders); kill switch cancels everything
* full audit log of submits, acks, rejects, cancels and fills
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable
from typing import Any

from .config import QuotingConfig
from .exchange import Exchange
from .models import EventKind, Fill, Liquidity, Order, OrderStatus, OrderType, Side
from .risk import RiskManager
from .strategies import MarketView, QuoteTarget, Strategy

Scheduler = Callable[[int, EventKind, Any], None]


class MarketMakerAgent:
    def __init__(
        self,
        owner: str,
        strategy: Strategy,
        qcfg: QuotingConfig,
        risk: RiskManager,
        exchange: Exchange,
        order_latency_ms: int,
        cancel_latency_ms: int,
        schedule: Scheduler,
    ) -> None:
        self.owner = owner
        self.strategy = strategy
        self.cfg = qcfg
        self.risk = risk
        self.exchange = exchange
        self.order_latency = order_latency_ms
        self.cancel_latency = cancel_latency_ms
        self.schedule = schedule
        self.open: dict[str, Order] = {}
        self.sent_at: dict[str, int] = {}
        self.pending_cancel: set[str] = set()
        self.log: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self.submitted_qty = 0
        self.last_target: QuoteTarget | None = None
        self.last_mid: float | None = None
        self._msg_times: deque[int] = deque()
        self._seq = 0
        exchange.subscribe(owner, self.on_fill)

    # ----------------------------------------------------------------- helpers
    def _purge(self) -> None:
        for oid in [oid for oid, o in self.open.items() if o.status.is_terminal]:
            del self.open[oid]
            self.pending_cancel.discard(oid)

    def open_qty(self, side: Side) -> int:
        return sum(
            o.remaining_qty
            for o in self.open.values()
            if o.side is side and not o.status.is_terminal
        )

    def active(self, side: Side) -> Order | None:
        for oid, o in self.open.items():
            if o.side is side and oid not in self.pending_cancel and not o.status.is_terminal:
                return o
        return None

    def resting_price(self, side: Side) -> int | None:
        prices = [
            o.price_ticks
            for o in self.open.values()
            if o.side is side and o.status is OrderStatus.RESTING and o.price_ticks is not None
        ]
        if not prices:
            return None
        return max(prices) if side is Side.BUY else min(prices)

    def _throttle_ok(self, now: int, n: int) -> bool:
        while self._msg_times and self._msg_times[0] <= now - 1000:
            self._msg_times.popleft()
        return len(self._msg_times) + n <= self.cfg.max_messages_per_sec

    def _record(self, now: int, event: str, order: Order | None = None, **extra: Any) -> None:
        row: dict[str, Any] = {"time_ms": now, "event": event}
        if order is not None:
            row.update(order_id=order.order_id, side=order.side.value,
                       price_ticks=order.price_ticks, quantity=order.quantity)
        row.update(extra)
        self.log.append(row)

    # ------------------------------------------------------------- decisions
    def on_wakeup(self, now: int, view: MarketView) -> QuoteTarget | None:
        self._purge()
        self.last_mid = view.mid if view.mid is not None else self.last_mid
        if self.risk.killed:
            self.cancel_all(now, "kill switch")
            self.last_target = None
            return None
        target = self.strategy.target(view)
        self.last_target = target
        self._reconcile(Side.BUY, target.bid_ticks, target.bid_qty, now, view.inventory)
        self._reconcile(Side.SELL, target.ask_ticks, target.ask_qty, now, view.inventory)
        return target

    def _reconcile(self, side: Side, price: int | None, qty: int, now: int, inv: int) -> None:
        current = self.active(side)
        if price is None or qty <= 0:
            if current is not None:
                self._cancel(current, now, forced=True, reason="side disabled")
            return
        if current is not None:
            assert current.price_ticks is not None
            if abs(current.price_ticks - price) < self.cfg.requote_threshold_ticks:
                return
            if now - self.sent_at[current.order_id] < self.cfg.min_quote_lifetime_ms:
                self.counts["lifetime_holds"] += 1
                return
            if not self._throttle_ok(now, 2):
                self.counts["throttled"] += 1
                return
            self._cancel(current, now, reason="requote")
        elif not self._throttle_ok(now, 1):
            self.counts["throttled"] += 1
            return
        self._submit(side, price, qty, now, inv)

    def _submit(self, side: Side, price: int, qty: int, now: int, inventory: int) -> None:
        decision = self.risk.check_order(
            side, price, qty, inventory, self.open_qty(Side.BUY), self.open_qty(Side.SELL)
        )
        order = Order(f"{self.owner}-{self._seq}", side, price, qty, self.owner, post_only=True)
        self._seq += 1
        if not decision.allowed:
            order.status = OrderStatus.REJECTED
            self.counts["risk_rejects"] += 1
            self._record(now, "risk_reject", order, reason=decision.reason)
            return
        self.open[order.order_id] = order
        self.sent_at[order.order_id] = now
        self._msg_times.append(now)
        self.counts["orders_submitted"] += 1
        self.submitted_qty += qty
        self._record(now, "submit", order)
        self.schedule(now + self.order_latency, EventKind.AGENT_ORDER, order)

    def _cancel(self, order: Order, now: int, forced: bool = False, reason: str = "") -> None:
        self.pending_cancel.add(order.order_id)
        self._msg_times.append(now)
        self.counts["cancels_sent"] += 1
        if forced:
            self.counts["forced_cancels"] += 1
        self._record(now, "cancel_request", order, reason=reason)
        self.schedule(now + self.cancel_latency, EventKind.AGENT_CANCEL, order.order_id)

    def cancel_all(self, now: int, reason: str) -> None:
        self._purge()
        for oid, o in list(self.open.items()):
            if oid not in self.pending_cancel and not o.status.is_terminal:
                self._cancel(o, now, forced=True, reason=reason)

    def flatten(self, now: int) -> None:
        """Send one market order closing current inventory (risk-driven, bypasses
        pre-trade checks and throttle). Fills arriving during latency may leave a residual."""
        inv = self.exchange.ledger(self.owner).inventory
        if inv == 0:
            return
        side = Side.SELL if inv > 0 else Side.BUY
        order = Order(f"{self.owner}-{self._seq}", side, None, abs(inv), self.owner,
                      order_type=OrderType.MARKET)
        self._seq += 1
        self.open[order.order_id] = order
        self.sent_at[order.order_id] = now
        self.counts["flatten_orders"] += 1
        self._record(now, "flatten_submit", order, reason="kill switch")
        self.schedule(now + self.order_latency, EventKind.AGENT_ORDER, order)

    # ------------------------------------------------------- exchange events
    def on_order_arrival(self, order: Order, now: int) -> None:
        res = self.exchange.submit(order, now)
        if res.accepted:
            self._record(now, "ack", order, status=order.status.value)
        else:
            self.counts["exchange_rejects"] += 1
            self._record(now, "exchange_reject", order, reason=res.reason)

    def on_cancel_arrival(self, order_id: str, now: int) -> None:
        cancelled = self.exchange.cancel(order_id, self.owner, now)
        order = self.open.get(order_id)
        if cancelled is not None:
            self.counts["cancels_confirmed"] += 1
            self._record(now, "cancel_ack", cancelled)
        else:
            self.counts["cancels_too_late"] += 1
            self._record(now, "cancel_reject", order, reason="not resting (filled/unknown)")

    def on_fill(self, fill: Fill) -> None:
        self.counts["fills"] += 1
        self.counts["maker_fills" if fill.liquidity is Liquidity.MAKER else "taker_fills"] += 1
        self._record(fill.timestamp, "fill", None, order_id=fill.order_id, side=fill.side.value,
                     price_ticks=fill.price_ticks, quantity=fill.quantity,
                     liquidity=fill.liquidity.value, fee=fill.fee)
