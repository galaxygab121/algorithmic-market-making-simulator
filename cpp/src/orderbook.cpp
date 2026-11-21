#include "orderbook.hpp"

namespace mm {

std::string OrderBook::validate(const Order& o) const {
    if (o.id.empty()) return "missing order id";
    if (seen_.contains(o.id)) return "duplicate order id";
    if (o.quantity <= 0) return "quantity must be a positive integer";
    if (o.type == OrderType::Limit) {
        if (o.price <= 0) return "limit price must be a positive integer number of ticks";
    } else if (o.post_only) {
        return "market orders cannot be post-only";
    }
    return {};
}

std::optional<std::int64_t> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<std::int64_t> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::int64_t OrderBook::level_quantity(Side side, std::int64_t price) const {
    std::int64_t total = 0;
    auto sum = [&](const auto& book) {
        auto it = book.find(price);
        if (it != book.end())
            for (const auto& o : it->second) total += o.remaining;
    };
    side == Side::Buy ? sum(bids_) : sum(asks_);
    return total;
}

SubmitResult OrderBook::submit(Order order) {
    SubmitResult res;
    order.remaining = order.quantity;
    if (auto reason = validate(order); !reason.empty()) {
        if (reason != "duplicate order id") seen_.insert(order.id);
        res.reason = std::move(reason);
        return res;
    }
    seen_.insert(order.id);
    if (order.post_only && order.type == OrderType::Limit) {
        if (order.side == Side::Buy) {
            if (auto a = best_ask(); a && order.price >= *a) {
                res.reason = "post-only order would cross";
                return res;
            }
        } else if (auto b = best_bid(); b && order.price <= *b) {
            res.reason = "post-only order would cross";
            return res;
        }
    }
    res.accepted = true;
    match(order, res);
    res.remaining = order.remaining;
    if (order.remaining > 0 && order.type == OrderType::Limit) {
        res.rested = true;
        rest(std::move(order));
    }
    return res;
}

void OrderBook::match(Order& o, SubmitResult& res) {
    auto run = [&](auto& book, auto crosses) {
        while (o.remaining > 0 && !book.empty()) {
            auto lvl = book.begin();
            if (o.type == OrderType::Limit && !crosses(o.price, lvl->first)) break;
            Level& q = lvl->second;
            while (!q.empty() && o.remaining > 0) {
                Order& maker = q.front();
                if (stp_ && maker.owner == o.owner) {
                    res.stp_cancelled.push_back(maker.id);
                    index_.erase(maker.id);
                    q.pop_front();
                    continue;
                }
                const std::int64_t qty = std::min(maker.remaining, o.remaining);
                maker.remaining -= qty;
                o.remaining -= qty;
                res.trades.push_back({next_trade_id_++, lvl->first, qty, maker.id, o.id});
                if (maker.remaining == 0) {
                    index_.erase(maker.id);
                    q.pop_front();
                }
            }
            if (q.empty()) book.erase(lvl);
        }
    };
    if (o.side == Side::Buy)
        run(asks_, [](std::int64_t limit, std::int64_t best) { return limit >= best; });
    else
        run(bids_, [](std::int64_t limit, std::int64_t best) { return limit <= best; });
}

void OrderBook::rest(Order&& o) {
    const Side side = o.side;
    const std::int64_t price = o.price;
    const std::string id = o.id;
    Level& q = side == Side::Buy ? bids_[price] : asks_[price];
    q.push_back(std::move(o));
    index_.emplace(id, Loc{side, price, std::prev(q.end())});
}

bool OrderBook::cancel(const std::string& id) {
    auto it = index_.find(id);
    if (it == index_.end()) return false;
    const Loc loc = it->second;
    index_.erase(it);
    auto erase_from = [&](auto& book) {
        auto lvl = book.find(loc.price);
        lvl->second.erase(loc.it);
        if (lvl->second.empty()) book.erase(lvl);
    };
    loc.side == Side::Buy ? erase_from(bids_) : erase_from(asks_);
    return true;
}

}  // namespace mm
