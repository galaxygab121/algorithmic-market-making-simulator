from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from marketmaker.config import ExchangeConfig, FeeConfig
from marketmaker.exchange import Exchange, FeeModel
from marketmaker.models import Fill, Ledger, Liquidity, Order, OrderStatus, OrderType, Side


def make_exchange(maker_bps: float = -0.2, taker_bps: float = 0.5,
                  tick: float = 0.01, mult: float = 10.0) -> Exchange:
    return Exchange(ExchangeConfig(tick_size=tick, contract_multiplier=mult,
                                   fees=FeeConfig(maker_fee_bps=maker_bps,
                                                  taker_fee_bps=taker_bps)))


def test_fee_signs_maker_rebate_taker_fee() -> None:
    ex = make_exchange()
    ex.submit(Order("m", Side.SELL, 10_000, 4, "maker"), now=0)
    ex.submit(Order("t", Side.BUY, None, 4, "taker", order_type=OrderType.MARKET), now=1)
    maker, taker = ex.ledger("maker"), ex.ledger("taker")
    notional = 10_000 * 4 * 0.1  # price_ticks * qty * tick_value = 4000.0
    assert maker.fees == pytest.approx(-notional * 0.2 / 1e4)  # rebate is negative fee
    assert taker.fees == pytest.approx(notional * 0.5 / 1e4)
    assert maker.fills[0].liquidity is Liquidity.MAKER
    assert taker.fills[0].liquidity is Liquidity.TAKER


def test_fee_model_zero_and_positive_maker_fee() -> None:
    fm = FeeModel(maker_fee_bps=0.1, taker_fee_bps=0.0, tick_value=1.0)
    assert fm.fee(100, 10, Liquidity.MAKER) == pytest.approx(1000 * 0.1 / 1e4)
    assert fm.fee(100, 10, Liquidity.TAKER) == 0.0


def test_cash_and_inventory_updated_on_fill() -> None:
    ex = make_exchange(maker_bps=0.0, taker_bps=0.0)
    ex.submit(Order("a", Side.SELL, 101, 3, "lp"), 0)
    ex.submit(Order("b", Side.BUY, None, 3, "mm", order_type=OrderType.MARKET), 5)
    mm = ex.ledger("mm")
    assert mm.inventory == 3
    assert mm.cash_ticks == -303
    assert ex.ledger("lp").inventory == -3
    assert ex.ledger("lp").cash_ticks == 303


def test_cancel_requires_owner_and_is_idempotent() -> None:
    ex = make_exchange()
    ex.submit(Order("x", Side.BUY, 99, 1, "alice"), 0)
    assert ex.cancel("x", "mallory", 1) is None
    assert ex.book.get("x") is not None
    assert ex.cancel("x", "alice", 2) is not None
    assert ex.cancel("x", "alice", 3) is None


def test_cancel_arriving_before_order_rejects_order() -> None:
    ex = make_exchange()
    assert ex.cancel("late", "mm", 10) is None
    o = Order("late", Side.BUY, 99, 1, "mm")
    res = ex.submit(o, 15)
    assert not res.accepted and res.reason == "cancelled before arrival"
    assert o.status is OrderStatus.REJECTED and ex.book.best_bid is None


def test_listener_receives_fills() -> None:
    ex = make_exchange()
    got: list[Fill] = []
    ex.subscribe("mm", got.append)
    ex.submit(Order("q", Side.BUY, 99, 2, "mm"), 0)
    ex.submit(Order("t", Side.SELL, None, 2, "x", order_type=OrderType.MARKET), 1)
    assert len(got) == 1 and got[0].liquidity is Liquidity.MAKER and got[0].quantity == 2


# ------------------------------------------------------------------ ledger / P&L
def fill(side: Side, px: int, qty: int, fee: float = 0.0, owner: str = "mm") -> Fill:
    return Fill(0, 0, "o", owner, side, px, qty, Liquidity.MAKER, fee)


def test_realized_pnl_round_trip() -> None:
    led = Ledger("mm", tick_value=0.5)
    led.apply_fill(fill(Side.BUY, 100, 10, fee=0.3))
    led.apply_fill(fill(Side.SELL, 104, 10, fee=0.3))
    assert led.inventory == 0
    assert led.position.realized_pnl == pytest.approx(4 * 10 * 0.5)
    assert led.equity(999) == pytest.approx(20.0 - 0.6)  # mark irrelevant when flat


def test_average_cost_and_unrealized() -> None:
    led = Ledger("mm", tick_value=1.0)
    led.apply_fill(fill(Side.BUY, 100, 2))
    led.apply_fill(fill(Side.BUY, 103, 1))
    assert led.position.avg_cost_ticks == pytest.approx(101.0)
    assert led.unrealized_pnl(105) == pytest.approx(3 * 4)
    led.apply_fill(fill(Side.SELL, 102, 1))
    assert led.position.realized_pnl == pytest.approx(1.0)
    assert led.position.avg_cost_ticks == pytest.approx(101.0)  # unchanged on reduce


def test_position_flip_long_to_short() -> None:
    led = Ledger("mm", tick_value=1.0)
    led.apply_fill(fill(Side.BUY, 100, 3))
    led.apply_fill(fill(Side.SELL, 98, 5))
    assert led.inventory == -2
    assert led.position.realized_pnl == pytest.approx(-6.0)
    assert led.position.avg_cost_ticks == 98.0
    assert led.unrealized_pnl(95) == pytest.approx(6.0)  # short gains when price falls


def test_ledger_rejects_wrong_owner_and_bad_qty() -> None:
    led = Ledger("mm", tick_value=1.0)
    with pytest.raises(ValueError):
        led.apply_fill(fill(Side.BUY, 100, 1, owner="other"))
    with pytest.raises(ValueError):
        led.apply_fill(fill(Side.BUY, 100, 0))


@settings(max_examples=300, deadline=None)
@given(
    st.lists(
        st.tuples(st.sampled_from([Side.BUY, Side.SELL]), st.integers(50, 150),
                  st.integers(1, 20), st.floats(-1, 1, allow_nan=False)),
        min_size=1, max_size=60),
    st.integers(50, 150),
)
def test_pnl_identity_holds_for_any_fill_sequence(
    fills: list[tuple[Side, int, int, float]], mark: int
) -> None:
    """equity = cash + q*mark - fees = realized + unrealized - fees, always."""
    led = Ledger("mm", tick_value=0.1)
    q = 0
    for side, px, qty, fee in fills:
        led.apply_fill(fill(side, px, qty, fee))
        q += side.sign * qty
    assert led.inventory == q
    lhs = led.equity(mark)
    rhs = led.position.realized_pnl + led.unrealized_pnl(mark) - led.fees
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_exchange_conserves_inventory_and_cash_across_owners() -> None:
    ex = make_exchange(maker_bps=0.0, taker_bps=0.0)
    ex.submit(Order("s1", Side.SELL, 101, 5, "a"), 0)
    ex.submit(Order("s2", Side.SELL, 102, 5, "b"), 0)
    ex.submit(Order("b1", Side.BUY, 102, 7, "c"), 1)
    ex.submit(Order("m1", Side.SELL, None, 4, "a", order_type=OrderType.MARKET), 2)
    assert sum(led.inventory for led in ex.ledgers.values()) == 0
    assert sum(led.cash_ticks for led in ex.ledgers.values()) == 0
    ex.book.check_invariants()
