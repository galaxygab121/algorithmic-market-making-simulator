"""Simulated exchange: order book + fee model + per-owner ledgers.

* Fees are charged per fill in basis points of notional
  (``price_ticks * qty * tick_value``). Maker fee may be negative (a rebate).
* Every trade produces two :class:`Fill` objects (maker and taker) that are applied to
  the owners' ledgers immediately, so cash and inventory are always consistent.
* Acknowledgements and fills are delivered to agents with zero latency (documented
  simplification); *inbound* order and cancel latency is modelled by the simulator.
* A cancel that arrives before its order is remembered; the order is then rejected on
  arrival ("cancelled before arrival"), so out-of-order latency can never orphan a quote.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .config import ExchangeConfig
from .models import Fill, Ledger, Liquidity, Order, OrderStatus, Side, Trade
from .orderbook import OrderBook, SubmitResult

FillListener = Callable[[Fill], None]


@dataclass(frozen=True)
class FeeModel:
    maker_fee_bps: float
    taker_fee_bps: float
    tick_value: float

    def notional(self, price_ticks: int, qty: int) -> float:
        return price_ticks * qty * self.tick_value

    def fee(self, price_ticks: int, qty: int, liquidity: Liquidity) -> float:
        bps = self.maker_fee_bps if liquidity is Liquidity.MAKER else self.taker_fee_bps
        return self.notional(price_ticks, qty) * bps / 10_000.0


class Exchange:
    def __init__(self, cfg: ExchangeConfig) -> None:
        self.cfg = cfg
        self.book = OrderBook(self_trade_prevention=cfg.self_trade_prevention)
        self.fees = FeeModel(cfg.fees.maker_fee_bps, cfg.fees.taker_fee_bps, cfg.tick_value)
        self.ledgers: dict[str, Ledger] = {}
        self.trades: list[Trade] = []
        self._listeners: dict[str, FillListener] = {}
        self._early_cancels: set[str] = set()

    def ledger(self, owner: str) -> Ledger:
        led = self.ledgers.get(owner)
        if led is None:
            led = Ledger(owner=owner, tick_value=self.cfg.tick_value)
            self.ledgers[owner] = led
        return led

    def subscribe(self, owner: str, listener: FillListener) -> None:
        self._listeners[owner] = listener

    def submit(self, order: Order, now: int) -> SubmitResult:
        if order.order_id in self._early_cancels:
            self._early_cancels.discard(order.order_id)
            order.status = OrderStatus.REJECTED
            return SubmitResult(False, order, reason="cancelled before arrival")
        order.timestamp = now
        result = self.book.submit(order)
        for trade in result.trades:
            self._settle(trade)
        return result

    def cancel(self, order_id: str, owner: str, now: int) -> Order | None:
        resting = self.book.get(order_id)
        if resting is None:
            if not self.book.has_seen(order_id):
                self._early_cancels.add(order_id)
            return None
        if resting.owner != owner:
            return None  # cannot cancel someone else's order
        return self.book.cancel(order_id)

    def _settle(self, trade: Trade) -> None:
        self.trades.append(trade)
        maker_side: Side = trade.aggressor_side.opposite
        for owner, order_id, side, liq in (
            (trade.maker_owner, trade.maker_order_id, maker_side, Liquidity.MAKER),
            (trade.taker_owner, trade.taker_order_id, trade.aggressor_side, Liquidity.TAKER),
        ):
            fill = Fill(
                trade_id=trade.trade_id,
                timestamp=trade.timestamp,
                order_id=order_id,
                owner=owner,
                side=side,
                price_ticks=trade.price_ticks,
                quantity=trade.quantity,
                liquidity=liq,
                fee=self.fees.fee(trade.price_ticks, trade.quantity, liq),
            )
            self.ledger(owner).apply_fill(fill)
            listener = self._listeners.get(owner)
            if listener is not None:
                listener(fill)
