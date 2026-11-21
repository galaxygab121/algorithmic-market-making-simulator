// mm_engine: replays a text command stream through the C++ order book.
//
// Input (one command per line):
//   L <id> <B|S> <price_ticks> <qty> <owner> <post_only 0|1>   limit order
//   M <id> <B|S> <qty> <owner>                                 market order (IOC)
//   C <id>                                                     cancel
// Output (--emit, default): one line per outcome, identical format to the Python
// reference in scripts/benchmark_cpp.py:
//   T <trade_id> <price> <qty> <maker_id> <taker_id>
//   R <id> <reason>        reject
//   S <id> <maker_id>      self-trade-prevention cancel of a resting order
//   X <id> | N <id>        cancel succeeded | not resting
// --bench <reps>: parse once, then time only the matching loop (fresh book per rep)
// and print "ns_per_command <value>" (median over reps). No I/O inside the timed region.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "orderbook.hpp"

namespace {

struct Command {
    char kind;
    mm::Order order;  // for L/M
    std::string id;   // for C
};

std::vector<Command> parse(std::istream& in) {
    std::vector<Command> cmds;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        Command c{};
        ss >> c.kind;
        if (c.kind == 'L' || c.kind == 'M') {
            char side = 0;
            int post = 0;
            mm::Order& o = c.order;
            ss >> o.id >> side;
            o.side = side == 'B' ? mm::Side::Buy : mm::Side::Sell;
            if (c.kind == 'L') {
                ss >> o.price >> o.quantity >> o.owner >> post;
                o.type = mm::OrderType::Limit;
                o.post_only = post != 0;
            } else {
                ss >> o.quantity >> o.owner;
                o.type = mm::OrderType::Market;
            }
        } else if (c.kind == 'C') {
            ss >> c.id;
        } else {
            throw std::runtime_error("bad command: " + line);
        }
        cmds.push_back(std::move(c));
    }
    return cmds;
}

}  // namespace

int main(int argc, char** argv) {
    std::string path;
    int bench_reps = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench" && i + 1 < argc) bench_reps = std::stoi(argv[++i]);
        else if (a == "--emit") bench_reps = 0;
        else path = a;
    }
    std::ifstream file;
    if (!path.empty()) file.open(path);
    std::istream& in = path.empty() ? std::cin : file;
    const auto cmds = parse(in);

    if (bench_reps > 0) {
        std::vector<double> samples;
        std::int64_t checksum = 0;
        for (int r = 0; r < bench_reps; ++r) {
            mm::OrderBook book(true);
            const auto t0 = std::chrono::steady_clock::now();
            for (const auto& c : cmds) {
                if (c.kind == 'C') checksum += book.cancel(c.id);
                else checksum += static_cast<std::int64_t>(book.submit(c.order).trades.size());
            }
            const auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() /
                              static_cast<double>(cmds.size()));
        }
        std::sort(samples.begin(), samples.end());
        std::cout << "commands " << cmds.size() << "\n"
                  << "ns_per_command " << samples[samples.size() / 2] << "\n"
                  << "checksum " << checksum << "\n";
        return 0;
    }

    mm::OrderBook book(true);
    std::ostringstream out;
    for (const auto& c : cmds) {
        if (c.kind == 'C') {
            out << (book.cancel(c.id) ? "X " : "N ") << c.id << "\n";
            continue;
        }
        auto res = book.submit(c.order);
        if (!res.accepted) {
            out << "R " << c.order.id << " " << res.reason << "\n";
            continue;
        }
        for (const auto& s : res.stp_cancelled) out << "S " << c.order.id << " " << s << "\n";
        for (const auto& t : res.trades)
            out << "T " << t.trade_id << " " << t.price << " " << t.quantity << " "
                << t.maker_id << " " << t.taker_id << "\n";
    }
    std::cout << out.str();
    return 0;
}
