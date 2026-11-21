// Price-time priority limit order book on integer ticks — C++20 port of
// src/marketmaker/orderbook.py with identical semantics:
//   * trades print at the resting (maker) price, best price first, FIFO within a level
//   * market orders are IOC (remainder cancelled), limit remainders rest
//   * post-only orders that would cross are rejected
//   * order IDs are single-use (duplicates rejected; invalid orders also burn the ID)
//   * optional self-trade prevention: cancel the resting order, keep matching
#pragma once

#include <cstdint>
#include <list>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mm {

enum class Side : std::uint8_t { Buy, Sell };
enum class OrderType : std::uint8_t { Limit, Market };

struct Order {
    std::string id;
    Side side{Side::Buy};
    std::int64_t price{0};  // ticks; ignored for market orders
    std::int64_t quantity{0};
    std::int64_t remaining{0};
    std::string owner;
    OrderType type{OrderType::Limit};
    bool post_only{false};
};

struct Trade {
    std::int64_t trade_id;
    std::int64_t price;
    std::int64_t quantity;
    std::string maker_id;
    std::string taker_id;
};

struct SubmitResult {
    bool accepted{false};
    std::string reason;  // empty if accepted
    std::vector<Trade> trades;
    std::vector<std::string> stp_cancelled;
    std::int64_t remaining{0};
    bool rested{false};
};

class OrderBook {
  public:
    explicit OrderBook(bool self_trade_prevention = true) : stp_(self_trade_prevention) {}

    SubmitResult submit(Order order);
    bool cancel(const std::string& id);

    [[nodiscard]] std::optional<std::int64_t> best_bid() const;
    [[nodiscard]] std::optional<std::int64_t> best_ask() const;
    [[nodiscard]] std::int64_t level_quantity(Side side, std::int64_t price) const;
    [[nodiscard]] std::size_t size() const { return index_.size(); }

  private:
    using Level = std::list<Order>;
    struct Loc {
        Side side;
        std::int64_t price;
        Level::iterator it;
    };

    std::string validate(const Order& o) const;
    void match(Order& o, SubmitResult& res);
    void rest(Order&& o);

    std::map<std::int64_t, Level, std::greater<>> bids_;  // best first
    std::map<std::int64_t, Level> asks_;                  // best first
    std::unordered_map<std::string, Loc> index_;
    std::unordered_set<std::string> seen_;
    std::int64_t next_trade_id_{1};
    bool stp_;
};

}  // namespace mm
