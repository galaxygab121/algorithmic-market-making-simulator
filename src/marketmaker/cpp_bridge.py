"""Optional bridge to the C++20 matching engine in ``cpp/`` (subprocess, no compilation
at import time). The simulator never depends on it; it exists for equivalence testing
and benchmarking of the order-book component in isolation.

Workloads are text command streams (see ``cpp/src/main.cpp`` for the format) derived
deterministically from a seeded synthetic market stream, with extra commands injected
to exercise post-only rejection, duplicate IDs and self-trade prevention.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from .config import SimulationConfig
from .marketdata import generate_market_data
from .models import EventKind, Order, OrderType, Side
from .orderbook import OrderBook

ROOT = Path(__file__).resolve().parents[2]
ENGINE_CANDIDATES = (
    ROOT / "cpp" / "build" / "mm_engine",
    ROOT / "cpp" / "build" / "Release" / "mm_engine.exe",
    ROOT / "cpp" / "build" / "mm_engine.exe",
)


def find_engine() -> Path | None:
    env = os.environ.get("MM_ENGINE")
    if env and Path(env).exists():
        return Path(env)
    for p in ENGINE_CANDIDATES:
        if p.exists():
            return p
    found = shutil.which("mm_engine")
    return Path(found) if found else None


def build_workload(cfg: SimulationConfig, seed: int | None = None) -> list[str]:
    stream = generate_market_data(cfg, seed)
    cmds: list[str] = []
    last_px = int(stream.fundamental_ticks[0])
    for i, ev in enumerate(stream.events):
        if ev.kind is EventKind.LP_LIMIT:
            assert ev.side is not None and ev.price_ticks is not None
            last_px = ev.price_ticks
            cmds.append(f"L {ev.order_id} {ev.side.value[0]} {ev.price_ticks} {ev.quantity} "
                        f"{ev.owner} 0")
        elif ev.kind is EventKind.LP_CANCEL:
            cmds.append(f"C {ev.order_id}")
        elif ev.kind is EventKind.NOISE_MARKET:
            assert ev.side is not None
            cmds.append(f"M {ev.order_id} {ev.side.value[0]} {ev.quantity} {ev.owner}")
        # Deterministic extra coverage
        if i % 50 == 7:  # post-only near the touch: some rest, some are rejected
            side = "B" if i % 100 == 7 else "S"
            cmds.append(f"L po{i} {side} {last_px} 3 MM 1")
        if i % 97 == 11:  # duplicate ID → reject
            cmds.append(f"L po{i - 4 if i > 4 else i} B {last_px} 1 MM 0")
        if i % 113 == 5:  # LP market order → self-trade prevention against LP quotes
            cmds.append(f"M stp{i} {'B' if i % 226 == 5 else 'S'} 4 LP")
    return cmds


def _parse(cmds: list[str]) -> list[tuple[str, ...]]:
    return [tuple(c.split()) for c in cmds]


def python_reference(cmds: list[str]) -> str:
    """Replay with the Python OrderBook and emit the C++ engine's output format."""
    book = OrderBook(self_trade_prevention=True)
    out: list[str] = []
    for parts in _parse(cmds):
        kind = parts[0]
        if kind == "C":
            out.append(f"{'X' if book.cancel(parts[1]) is not None else 'N'} {parts[1]}")
            continue
        order = _make_order(parts)
        res = book.submit(order)
        if not res.accepted:
            out.append(f"R {order.order_id} {res.reason}")
            continue
        out.extend(f"S {order.order_id} {s.order_id}" for s in res.stp_cancelled)
        out.extend(f"T {t.trade_id} {t.price_ticks} {t.quantity} {t.maker_order_id} "
                   f"{t.taker_order_id}" for t in res.trades)
    return "\n".join(out) + ("\n" if out else "")


def _make_order(parts: tuple[str, ...]) -> Order:
    side = Side.BUY if parts[2] == "B" else Side.SELL
    if parts[0] == "L":
        return Order(parts[1], side, int(parts[3]), int(parts[4]), parts[5],
                     post_only=parts[6] == "1")
    return Order(parts[1], side, None, int(parts[3]), parts[4], order_type=OrderType.MARKET)


def run_engine(cmds: list[str], engine: Path, workdir: Path, bench_reps: int = 0) -> str:
    path = workdir / "workload.txt"
    path.write_text("\n".join(cmds) + "\n", encoding="utf-8")
    args = [str(engine), str(path)]
    if bench_reps:
        args += ["--bench", str(bench_reps)]
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def python_ns_per_command(cmds: list[str], reps: int = 5) -> float:
    """Median ns/command for the Python book; parsing happens outside the timed loop,
    Order construction is inside (the C++ engine also copies an Order per submit)."""
    parsed = _parse(cmds)
    samples = []
    for _ in range(reps):
        book = OrderBook(self_trade_prevention=True)
        t0 = time.perf_counter_ns()
        for parts in parsed:
            if parts[0] == "C":
                book.cancel(parts[1])
            else:
                book.submit(_make_order(parts))
        samples.append((time.perf_counter_ns() - t0) / len(parsed))
    samples.sort()
    return samples[len(samples) // 2]
