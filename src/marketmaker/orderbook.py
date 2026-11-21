"""Price-time priority limit order book on integer ticks.

* Bids and asks are kept as ``price -> deque[Order]`` (FIFO within a level) plus an
  ascending sorted list of active prices per side.
* Incoming market orders and marketable limit orders cross the book best-price-first,
  oldest-first. Trades always print at the **resting (maker) order's price**.
* Market orders are immediate-or-cancel: an unfilled remainder is cancelled, never rested.
* Limit remainders rest at their limit price.
* Post-only orders that would immediately cross are rejected (not repriced).
* Order IDs are single-use: re-submitting an ID that the book has already seen is
  rejected, so retries are idempotent and cannot double an exposure.
* Optional self-trade prevention (cancel-resting): an incoming order never trades
  against a resting order of the same owner; the resting order is cancelled instead.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from .models import Order, OrderStatus, OrderType, Side, Trade


@dataclass(slots=True)
class SubmitResult:
    accepted: bool
    order: Order
    trades: list[Trade] = field(default_factory=list)
    reason: str = ""
    stp_cancelled: list[Order] = field(default_factory=list)


class OrderBook:
    def __init__(self, self_trade_prevention: bool = True) -> None:
        self._levels: dict[Side, dict[int, deque[Order]]] = {Side.BUY: {}, Side.SELL: {}}
        self._prices: dict[Side, list[int]] = {Side.BUY: [], Side.SELL: []}  # ascending
        self._resting: dict[str, Order] = {}
        self._seen_ids: set[str] = set()
        self._next_trade_id = 1
        self.self_trade_prevention = self_trade_prevention

    # ------------------------------------------------------------------ queries
    def best(self, side: Side) -> int | None:
        prices = self._prices[side]
        if not prices:
            return None
        return prices[-1] if side is Side.BUY else prices[0]

    @property
    def best_bid(self) -> int | None:
        return self.best(Side.BUY)

    @property
    def best_ask(self) -> int | None:
        return self.best(Side.SELL)

    def prices_best_first(self, side: Side) -> Iterator[int]:
        prices = self._prices[side]
        return reversed(prices) if side is Side.BUY else iter(prices)

    def level(self, side: Side, price: int) -> list[Order]:
        return list(self._levels[side].get(price, ()))

    def level_quantity(self, side: Side, price: int) -> int:
        return sum(o.remaining_qty for o in self._levels[side].get(price, ()))

    def depth(self, side: Side, levels: int = 10) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for price in self.prices_best_first(side):
            out.append((price, self.level_quantity(side, price)))
            if len(out) >= levels:
                break
        return out

    def best_excluding(self, side: Side, owner: str) -> int | None:
        """Best price on ``side`` ignoring ``owner``'s own resting orders."""
        levels = self._levels[side]
        for price in self.prices_best_first(side):
            if any(o.owner != owner for o in levels[price]):
                return price
        return None

    def get(self, order_id: str) -> Order | None:
        return self._resting.get(order_id)

    def __contains__(self, order_id: object) -> bool:
        return order_id in self._resting

    def __len__(self) -> int:
        return len(self._resting)

    def has_seen(self, order_id: str) -> bool:
        return order_id in self._seen_ids

    def resting_orders(self, owner: str | None = None) -> list[Order]:
        return [o for o in self._resting.values() if owner is None or o.owner == owner]

    # --------------------------------------------------------------- mutations
    def _validate(self, order: Order) -> str:
        if not order.order_id:
            return "missing order id"
        if order.order_id in self._seen_ids:
            return "duplicate order id"
        if not isinstance(order.quantity, int) or order.quantity <= 0:
            return "quantity must be a positive integer"
        if order.remaining_qty != order.quantity:
            return "new order must have remaining_qty == quantity"
        if order.order_type is OrderType.LIMIT:
            if not isinstance(order.price_ticks, int) or order.price_ticks <= 0:
                return "limit price must be a positive integer number of ticks"
        else:
            if order.price_ticks is not None:
                return "market orders must not specify a price"
            if order.post_only:
                return "market orders cannot be post-only"
        return ""

    def _crosses(self, side: Side, limit: int, opposite_best: int) -> bool:
        return limit >= opposite_best if side is Side.BUY else limit <= opposite_best

    def submit(self, order: Order) -> SubmitResult:
        reason = self._validate(order)
        if reason:
            if reason != "duplicate order id":
                self._seen_ids.add(order.order_id)
            order.status = OrderStatus.REJECTED
            return SubmitResult(False, order, reason=reason)
        self._seen_ids.add(order.order_id)

        if order.post_only and order.price_ticks is not None:
            opp_best = self.best(order.side.opposite)
            if opp_best is not None and self._crosses(order.side, order.price_ticks, opp_best):
                order.status = OrderStatus.REJECTED
                return SubmitResult(False, order, reason="post-only order would cross")

        trades, stp = self._match(order)
        if order.remaining_qty == 0:
            order.status = OrderStatus.FILLED
        elif order.order_type is OrderType.LIMIT:
            self._rest(order)
            order.status = OrderStatus.RESTING
        else:
            order.status = OrderStatus.CANCELLED  # IOC remainder (incl. empty book)
        return SubmitResult(True, order, trades=trades, stp_cancelled=stp)

    def _match(self, order: Order) -> tuple[list[Trade], list[Order]]:
        trades: list[Trade] = []
        stp: list[Order] = []
        opp = order.side.opposite
        prices = self._prices[opp]
        levels = self._levels[opp]
        while order.remaining_qty > 0 and prices:
            best = prices[0] if opp is Side.SELL else prices[-1]
            if order.order_type is OrderType.LIMIT:
                assert order.price_ticks is not None
                if not self._crosses(order.side, order.price_ticks, best):
                    break
            queue = levels[best]
            while queue and order.remaining_qty > 0:
                maker = queue[0]
                if self.self_trade_prevention and maker.owner == order.owner:
                    queue.popleft()
                    del self._resting[maker.order_id]
                    maker.status = OrderStatus.CANCELLED
                    stp.append(maker)
                    continue
                qty = min(maker.remaining_qty, order.remaining_qty)
                maker.remaining_qty -= qty
                order.remaining_qty -= qty
                trades.append(
                    Trade(
                        trade_id=self._next_trade_id,
                        timestamp=order.timestamp,
                        price_ticks=best,
                        quantity=qty,
                        maker_order_id=maker.order_id,
                        taker_order_id=order.order_id,
                        maker_owner=maker.owner,
                        taker_owner=order.owner,
                        aggressor_side=order.side,
                    )
                )
                self._next_trade_id += 1
                if maker.remaining_qty == 0:
                    queue.popleft()
                    del self._resting[maker.order_id]
                    maker.status = OrderStatus.FILLED
            if not queue:
                del levels[best]
                if opp is Side.SELL:
                    prices.pop(0)
                else:
                    prices.pop()
        return trades, stp

    def _rest(self, order: Order) -> None:
        assert order.price_ticks is not None
        levels = self._levels[order.side]
        queue = levels.get(order.price_ticks)
        if queue is None:
            queue = deque()
            levels[order.price_ticks] = queue
            bisect.insort(self._prices[order.side], order.price_ticks)
        queue.append(order)
        self._resting[order.order_id] = order

    def cancel(self, order_id: str) -> Order | None:
        """Cancel a resting order. Returns the order, or ``None`` if it is not resting
        (unknown, already filled or already cancelled) — cancelling is idempotent."""
        order = self._resting.pop(order_id, None)
        if order is None:
            return None
        assert order.price_ticks is not None
        levels = self._levels[order.side]
        queue = levels[order.price_ticks]
        queue.remove(order)
        if not queue:
            del levels[order.price_ticks]
            prices = self._prices[order.side]
            idx = bisect.bisect_left(prices, order.price_ticks)
            prices.pop(idx)
        order.status = OrderStatus.CANCELLED
        return order

    # -------------------------------------------------------------- invariants
    def check_invariants(self) -> None:
        """Raise AssertionError if the book is internally inconsistent."""
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None:
            assert bid < ask, f"crossed book after matching: bid {bid} >= ask {ask}"
        count = 0
        for side in (Side.BUY, Side.SELL):
            levels = self._levels[side]
            prices = self._prices[side]
            assert prices == sorted(prices), "price list not sorted"
            assert set(prices) == set(levels), "price list / level map mismatch"
            for price, queue in levels.items():
                assert queue, f"empty level left at {price}"
                for o in queue:
                    assert o.remaining_qty > 0, "non-positive remaining qty on book"
                    assert o.remaining_qty <= o.quantity
                    assert o.price_ticks == price and o.side is side
                    assert self._resting.get(o.order_id) is o
                    count += 1
        assert count == len(self._resting), "resting index out of sync"
