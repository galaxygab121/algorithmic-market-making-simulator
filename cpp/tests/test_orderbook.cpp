#include <catch2/catch_test_macros.hpp>

#include "orderbook.hpp"

using mm::Order;
using mm::OrderBook;
using mm::OrderType;
using mm::Side;

namespace {
Order lim(const std::string& id, Side s, std::int64_t px, std::int64_t q,
          const std::string& owner = "a", bool post = false) {
    return Order{id, s, px, q, q, owner, OrderType::Limit, post};
}
Order mkt(const std::string& id, Side s, std::int64_t q, const std::string& owner = "t") {
    return Order{id, s, 0, q, q, owner, OrderType::Market, false};
}
}  // namespace

TEST_CASE("FIFO within a price level") {
    OrderBook b;
    b.submit(lim("1", Side::Sell, 101, 3, "first"));
    b.submit(lim("2", Side::Sell, 101, 3, "second"));
    auto r = b.submit(mkt("3", Side::Buy, 4));
    REQUIRE(r.trades.size() == 2);
    CHECK(r.trades[0].maker_id == "1");
    CHECK(r.trades[0].quantity == 3);
    CHECK(r.trades[1].maker_id == "2");
    CHECK(b.level_quantity(Side::Sell, 101) == 2);
}

TEST_CASE("price priority and maker-price execution, remainder rests") {
    OrderBook b;
    b.submit(lim("a", Side::Sell, 102, 2));
    b.submit(lim("b", Side::Sell, 101, 2));
    auto r = b.submit(lim("c", Side::Buy, 103, 5, "x"));
    REQUIRE(r.trades.size() == 2);
    CHECK(r.trades[0].price == 101);
    CHECK(r.trades[1].price == 102);
    CHECK(r.rested);
    CHECK(*b.best_bid() == 103);
    CHECK_FALSE(b.best_ask().has_value());
}

TEST_CASE("market order IOC on thin and empty book") {
    OrderBook b;
    auto empty = b.submit(mkt("m0", Side::Buy, 5));
    CHECK(empty.accepted);
    CHECK(empty.trades.empty());
    b.submit(lim("l", Side::Buy, 99, 3));
    auto r = b.submit(mkt("m1", Side::Sell, 10));
    CHECK(r.remaining == 7);
    CHECK_FALSE(r.rested);
    CHECK(b.size() == 0);
}

TEST_CASE("rejects invalid, duplicate and crossing post-only orders") {
    OrderBook b;
    CHECK_FALSE(b.submit(lim("q0", Side::Buy, 99, 0)).accepted);
    CHECK_FALSE(b.submit(lim("p0", Side::Buy, -1, 5)).accepted);
    CHECK(b.submit(lim("d", Side::Sell, 101, 5)).accepted);
    CHECK(b.submit(lim("d", Side::Sell, 101, 5)).reason == "duplicate order id");
    CHECK(b.submit(lim("po", Side::Buy, 101, 1, "a", true)).reason ==
          "post-only order would cross");
    CHECK(b.level_quantity(Side::Sell, 101) == 5);
}

TEST_CASE("cancel is idempotent and preserves FIFO of the rest") {
    OrderBook b;
    b.submit(lim("a", Side::Sell, 101, 1, "a"));
    b.submit(lim("b", Side::Sell, 101, 1, "b"));
    b.submit(lim("c", Side::Sell, 101, 1, "c"));
    CHECK(b.cancel("b"));
    CHECK_FALSE(b.cancel("b"));
    auto r = b.submit(mkt("m", Side::Buy, 2));
    CHECK(r.trades[0].maker_id == "a");
    CHECK(r.trades[1].maker_id == "c");
}

TEST_CASE("self-trade prevention cancels resting") {
    OrderBook b(true);
    b.submit(lim("own", Side::Sell, 101, 5, "mm"));
    b.submit(lim("oth", Side::Sell, 102, 5, "lp"));
    auto r = b.submit(mkt("m", Side::Buy, 3, "mm"));
    REQUIRE(r.stp_cancelled.size() == 1);
    CHECK(r.trades[0].price == 102);
}
