"""Core domain entities: orders, trades, fills, quotes, events, positions and the ledger.

Conventions
-----------
* Prices are integer **ticks** and quantities are integer **units**; matching never
  touches floating point.
* ``tick_value`` = currency value of a one-tick move for one unit
  (``tick_size * contract_multiplier``). Cash, fees and P&L are reported in currency.
* Cash is kept as an exact integer number of tick-units (``cash_ticks``) and converted
  to currency only when reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """+1 for buys (inventory increases), -1 for sells."""
        return 1 if self is Side.BUY else -1


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"  # immediate-or-cancel against the book; never rests


class OrderStatus(StrEnum):
    PENDING = "PENDING"  # created by an agent, not yet arrived at the exchange
    RESTING = "RESTING"  # on the book (possibly partially filled)
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


class Liquidity(StrEnum):
    MAKER = "MAKER"
    TAKER = "TAKER"


class EventKind(IntEnum):
    """Event types. The integer value is the deterministic tie-break priority for
    events that share a timestamp (lower value is processed first)."""

    AGENT_CANCEL = 0  # market maker cancel request arriving at the exchange
    AGENT_ORDER = 1  # market maker new order arriving at the exchange
    LP_CANCEL = 2  # background liquidity provider cancel
    LP_LIMIT = 3  # background liquidity provider limit order
    NOISE_MARKET = 4  # noise / informed trader market order
    STRATEGY_WAKEUP = 5  # strategy decision point (sees everything at t above)
    SAMPLE = 6  # accounting snapshot / risk monitor


@dataclass(slots=True)
class Order:
    order_id: str
    side: Side
    price_ticks: int | None  # None for market orders
    quantity: int
    owner: str
    timestamp: int = 0  # exchange arrival time (ms); set by the exchange
    order_type: OrderType = OrderType.LIMIT
    post_only: bool = False
    remaining_qty: int = -1
    status: OrderStatus = OrderStatus.PENDING

    def __post_init__(self) -> None:
        if self.remaining_qty == -1:
            self.remaining_qty = self.quantity

    @property
    def filled_qty(self) -> int:
        return self.quantity - self.remaining_qty


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: int
    timestamp: int
    price_ticks: int
    quantity: int
    maker_order_id: str
    taker_order_id: str
    maker_owner: str
    taker_owner: str
    aggressor_side: Side  # side of the taker

    @property
    def buyer(self) -> str:
        return self.taker_owner if self.aggressor_side is Side.BUY else self.maker_owner

    @property
    def seller(self) -> str:
        return self.maker_owner if self.aggressor_side is Side.BUY else self.taker_owner


@dataclass(frozen=True, slots=True)
class Fill:
    """One side of a trade, from the perspective of a single owner."""

    trade_id: int
    timestamp: int
    order_id: str
    owner: str
    side: Side
    price_ticks: int
    quantity: int
    liquidity: Liquidity
    fee: float  # currency; positive = paid, negative = rebate received


@dataclass(frozen=True, slots=True)
class Quote:
    timestamp: int
    bid_ticks: int | None
    ask_ticks: int | None
    bid_qty: int
    ask_qty: int

    @property
    def spread_ticks(self) -> int | None:
        if self.bid_ticks is None or self.ask_ticks is None:
            return None
        return self.ask_ticks - self.bid_ticks


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """An exogenous market event. Immutable so one stream can be replayed against
    several independent exchanges (one per strategy) for a fair comparison."""

    timestamp: int
    kind: EventKind
    order_id: str = ""
    side: Side | None = None
    price_ticks: int | None = None
    quantity: int = 0
    owner: str = ""


@dataclass(slots=True)
class Position:
    quantity: int = 0
    avg_cost_ticks: float = 0.0  # average entry price of the *open* position
    realized_pnl: float = 0.0  # currency, excluding fees


@dataclass
class Ledger:
    """Per-owner cash / inventory / P&L accounting.

    Identity maintained (starting from zero cash and inventory)::

        equity(mark) = cash + inventory * mark * tick_value - fees
                     = realized_pnl + unrealized_pnl(mark) - fees

    Realized P&L uses the average-cost method; the position may flip sign in one fill.
    """

    owner: str
    tick_value: float
    cash_ticks: int = 0  # sum(sell px * qty) - sum(buy px * qty), exact integer
    fees: float = 0.0  # net fees paid (rebates reduce this, may go negative)
    position: Position = field(default_factory=Position)
    fills: list[Fill] = field(default_factory=list)
    volume: int = 0

    @property
    def inventory(self) -> int:
        return self.position.quantity

    @property
    def cash(self) -> float:
        return self.cash_ticks * self.tick_value

    def apply_fill(self, fill: Fill) -> None:
        if fill.owner != self.owner:
            raise ValueError(f"fill for {fill.owner!r} applied to ledger of {self.owner!r}")
        if fill.quantity <= 0:
            raise ValueError("fill quantity must be positive")
        signed = fill.quantity * fill.side.sign
        self.cash_ticks -= signed * fill.price_ticks
        self.fees += fill.fee
        self.volume += fill.quantity
        self.fills.append(fill)

        pos = self.position
        q = pos.quantity
        new_q = q + signed
        if q == 0 or (q > 0) == (signed > 0):
            # Opening or increasing: blend the average cost.
            pos.avg_cost_ticks = (
                pos.avg_cost_ticks * abs(q) + fill.price_ticks * abs(signed)
            ) / abs(new_q)
        else:
            closing = min(abs(q), abs(signed))
            direction = 1 if q > 0 else -1
            pos.realized_pnl += (
                closing * (fill.price_ticks - pos.avg_cost_ticks) * direction * self.tick_value
            )
            if new_q == 0:
                pos.avg_cost_ticks = 0.0
            elif (new_q > 0) != (q > 0):
                pos.avg_cost_ticks = float(fill.price_ticks)  # flipped: remainder is new
        pos.quantity = new_q

    def unrealized_pnl(self, mark_ticks: float) -> float:
        q = self.position.quantity
        return q * (mark_ticks - self.position.avg_cost_ticks) * self.tick_value if q else 0.0

    def equity(self, mark_ticks: float) -> float:
        return (self.cash_ticks + self.position.quantity * mark_ticks) * self.tick_value - self.fees
