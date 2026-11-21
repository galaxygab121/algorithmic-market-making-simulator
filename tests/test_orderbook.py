from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from marketmaker.models import Order, OrderStatus, OrderType, Side
from marketmaker.orderbook import OrderBook

_ids = iter(range(10**9))


def lim(side: Side, price: int, qty: int, owner: str = "a", post_only: bool = False) -> Order:
    return Order(f"o{next(_ids)}", side, price, qty, owner, post_only=post_only)


def mkt(side: Side, qty: int, owner: str = "t") -> Order:
    return Order(f"o{next(_ids)}", side, None, qty, owner, order_type=OrderType.MARKET)


def test_resting_orders_and_best_prices() -> None:
    book = OrderBook()
    book.submit(lim(Side.BUY, 99, 5))
    book.submit(lim(Side.BUY, 98, 5))
    book.submit(lim(Side.SELL, 101, 5))
    book.submit(lim(Side.SELL, 102, 5))
    assert book.best_bid == 99
    assert book.best_ask == 101
    assert book.depth(Side.BUY) == [(99, 5), (98, 5)]
    assert book.depth(Side.SELL) == [(101, 5), (102, 5)]
    book.check_invariants()


def test_fifo_time_priority_within_level() -> None:
    book = OrderBook()
    first = lim(Side.SELL, 101, 3, owner="first")
    second = lim(Side.SELL, 101, 3, owner="second")
    book.submit(first)
    book.submit(second)
    res = book.submit(mkt(Side.BUY, 4))
    assert [t.maker_owner for t in res.trades] == ["first", "second"]
    assert [t.quantity for t in res.trades] == [3, 1]
    assert first.status is OrderStatus.FILLED
    assert second.remaining_qty == 2 and second.status is OrderStatus.RESTING


def test_price_priority_beats_time_priority() -> None:
    book = OrderBook()
    book.submit(lim(Side.SELL, 102, 5, owner="early_worse"))
    book.submit(lim(Side.SELL, 101, 5, owner="late_better"))
    res = book.submit(mkt(Side.BUY, 5))
    assert res.trades[0].maker_owner == "late_better"
    assert res.trades[0].price_ticks == 101


def test_marketable_limit_crosses_at_maker_prices_and_rests_remainder() -> None:
    book = OrderBook()
    book.submit(lim(Side.SELL, 101, 2))
    book.submit(lim(Side.SELL, 102, 2))
    book.submit(lim(Side.SELL, 104, 2))
    incoming = lim(Side.BUY, 103, 10, owner="b")
    res = book.submit(incoming)
    assert [(t.price_ticks, t.quantity) for t in res.trades] == [(101, 2), (102, 2)]
    assert incoming.remaining_qty == 6
    assert incoming.status is OrderStatus.RESTING
    assert book.best_bid == 103 and book.best_ask == 104
    book.check_invariants()


def test_market_order_partial_fill_remainder_cancelled() -> None:
    book = OrderBook()
    book.submit(lim(Side.BUY, 99, 3))
    order = mkt(Side.SELL, 10)
    res = book.submit(order)
    assert res.accepted
    assert sum(t.quantity for t in res.trades) == 3
    assert order.remaining_qty == 7
    assert order.status is OrderStatus.CANCELLED
    assert book.best_bid is None  # IOC remainder never rests
    assert len(book) == 0


def test_market_order_on_empty_book() -> None:
    book = OrderBook()
    order = mkt(Side.BUY, 5)
    res = book.submit(order)
    assert res.accepted and res.trades == []
    assert order.status is OrderStatus.CANCELLED


def test_non_marketable_limit_does_not_trade() -> None:
    book = OrderBook()
    book.submit(lim(Side.SELL, 101, 5))
    res = book.submit(lim(Side.BUY, 100, 5))
    assert res.trades == []
    assert book.best_bid == 100 and book.best_ask == 101


def test_cancel_and_idempotent_cancel() -> None:
    book = OrderBook()
    o = lim(Side.BUY, 99, 5)
    book.submit(o)
    assert book.cancel(o.order_id) is o
    assert o.status is OrderStatus.CANCELLED
    assert book.best_bid is None
    assert book.cancel(o.order_id) is None  # second cancel is a harmless no-op
    assert book.cancel("never-existed") is None
    book.check_invariants()


def test_cancel_middle_of_queue_preserves_fifo() -> None:
    book = OrderBook()
    a, b, c = (lim(Side.SELL, 101, 1, owner=n) for n in "abc")
    for o in (a, b, c):
        book.submit(o)
    book.cancel(b.order_id)
    res = book.submit(mkt(Side.BUY, 2))
    assert [t.maker_owner for t in res.trades] == ["a", "c"]


def test_cannot_cancel_filled_order() -> None:
    book = OrderBook()
    o = lim(Side.SELL, 101, 2)
    book.submit(o)
    book.submit(mkt(Side.BUY, 2))
    assert o.status is OrderStatus.FILLED
    assert book.cancel(o.order_id) is None


def test_duplicate_order_id_rejected_idempotently() -> None:
    book = OrderBook()
    o1 = Order("dup", Side.BUY, 99, 5, "a")
    o2 = Order("dup", Side.BUY, 99, 5, "a")
    assert book.submit(o1).accepted
    res = book.submit(o2)
    assert not res.accepted and res.reason == "duplicate order id"
    assert book.level_quantity(Side.BUY, 99) == 5  # exposure not doubled
    assert o1.status is OrderStatus.RESTING  # original untouched


def test_reused_id_after_cancel_still_rejected() -> None:
    book = OrderBook()
    o1 = Order("x", Side.BUY, 99, 5, "a")
    book.submit(o1)
    book.cancel("x")
    assert not book.submit(Order("x", Side.BUY, 99, 5, "a")).accepted


def test_invalid_inputs_rejected() -> None:
    book = OrderBook()
    bad = [
        Order("n1", Side.BUY, 99, 0, "a"),
        Order("n2", Side.BUY, 99, -3, "a"),
        Order("n3", Side.BUY, -1, 5, "a"),
        Order("n4", Side.BUY, 0, 5, "a"),
        Order("n5", Side.BUY, None, 5, "a"),  # limit without price
        Order("n6", Side.BUY, 99, 5, "a", order_type=OrderType.MARKET),  # market w/ price
        Order("n7", Side.BUY, None, 5, "a", order_type=OrderType.MARKET, post_only=True),
        Order("", Side.BUY, 99, 5, "a"),
    ]
    for o in bad:
        res = book.submit(o)
        assert not res.accepted, o
        assert o.status is OrderStatus.REJECTED
    assert len(book) == 0


def test_post_only_rejected_when_it_would_cross() -> None:
    book = OrderBook()
    book.submit(lim(Side.SELL, 101, 5))
    res = book.submit(lim(Side.BUY, 101, 5, post_only=True))
    assert not res.accepted and "post-only" in res.reason
    assert book.submit(lim(Side.BUY, 100, 5, post_only=True)).accepted
    assert book.best_bid == 100


def test_self_trade_prevention_cancels_resting() -> None:
    book = OrderBook(self_trade_prevention=True)
    own = lim(Side.SELL, 101, 5, owner="mm")
    other = lim(Side.SELL, 102, 5, owner="lp")
    book.submit(own)
    book.submit(other)
    res = book.submit(mkt(Side.BUY, 3, owner="mm"))
    assert res.stp_cancelled == [own] and own.status is OrderStatus.CANCELLED
    assert res.trades[0].maker_owner == "lp" and res.trades[0].price_ticks == 102


def test_best_excluding_owner() -> None:
    book = OrderBook()
    book.submit(lim(Side.BUY, 100, 1, owner="mm"))
    book.submit(lim(Side.BUY, 99, 1, owner="lp"))
    assert book.best_bid == 100
    assert book.best_excluding(Side.BUY, "mm") == 99
    assert book.best_excluding(Side.SELL, "mm") is None


def test_trade_ids_unique_and_increasing() -> None:
    book = OrderBook()
    for p in (101, 102, 103):
        book.submit(lim(Side.SELL, p, 1))
    res = book.submit(mkt(Side.BUY, 3))
    ids = [t.trade_id for t in res.trades]
    assert ids == sorted(ids) and len(set(ids)) == 3


# --------------------------------------------------------------------- property tests
order_strategy = st.tuples(
    st.sampled_from(["limit", "market", "cancel"]),
    st.sampled_from([Side.BUY, Side.SELL]),
    st.integers(min_value=95, max_value=105),
    st.integers(min_value=1, max_value=10),
    st.sampled_from(["a", "b", "c"]),
)


@settings(max_examples=200, deadline=None)
@given(st.lists(order_strategy, min_size=1, max_size=80))
def test_random_order_flow_invariants(ops: list[tuple[str, Side, int, int, str]]) -> None:
    book = OrderBook(self_trade_prevention=False)
    submitted: list[Order] = []
    bought: dict[str, int] = {}
    sold: dict[str, int] = {}
    for i, (kind, side, price, qty, owner) in enumerate(ops):
        if kind == "cancel":
            if submitted:
                book.cancel(submitted[price % len(submitted)].order_id)
        else:
            o = (
                Order(f"p{i}", side, price, qty, owner)
                if kind == "limit"
                else Order(f"p{i}", side, None, qty, owner, order_type=OrderType.MARKET)
            )
            submitted.append(o)
            res = book.submit(o)
            for t in res.trades:
                assert t.quantity > 0
                bought[t.buyer] = bought.get(t.buyer, 0) + t.quantity
                sold[t.seller] = sold.get(t.seller, 0) + t.quantity
        book.check_invariants()  # includes best bid < best ask, no empty levels
        for o in submitted:
            assert 0 <= o.remaining_qty <= o.quantity
    # Every unit bought was sold by someone: buyer/seller quantity symmetry.
    assert sum(bought.values()) == sum(sold.values())
    # Conservation: filled quantity across all orders = 2 x traded volume.
    traded = sum(bought.values())
    assert sum(o.filled_qty for o in submitted) == 2 * traded
