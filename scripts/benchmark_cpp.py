"""Benchmark the Python vs C++ order book on an *identical* command workload.

Methodology
-----------
* Workload: ``build_workload`` on examples/baseline.yaml with ``--steps`` steps (seeded),
  i.e. LP limit orders, LP cancels and taker market orders from the synthetic stream plus
  injected post-only / duplicate-ID / self-trade commands.
* Equivalence is verified first (outputs must be byte-identical), else the run aborts.
* Timed region: only the submit/cancel loop on a fresh book. Parsing and I/O are
  excluded on both sides. Python builds an ``Order`` per command inside the loop; C++
  copies an ``Order`` per submit. Reported value: median ns/command over ``--reps``.
* C++ built with CMake ``Release`` (compiler default optimisation, ``-O3`` for clang/gcc).

    uv run python scripts/benchmark_cpp.py --steps 20000 --reps 7
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marketmaker.config import load_config  # noqa: E402
from marketmaker.cpp_bridge import (  # noqa: E402
    build_workload,
    find_engine,
    python_ns_per_command,
    python_reference,
    run_engine,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    engine = find_engine()
    if engine is None:
        print("C++ engine not built: cmake -S cpp -B cpp/build && cmake --build cpp/build")
        return 1
    cfg = load_config(ROOT / "examples" / "baseline.yaml").with_overrides(steps=args.steps)
    cmds = build_workload(cfg, args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        if run_engine(cmds, engine, Path(tmp)) != python_reference(cmds):
            print("ERROR: Python and C++ outputs differ; refusing to benchmark")
            return 2
        cpp_out = run_engine(cmds, engine, Path(tmp), bench_reps=args.reps)
    cpp_ns = float(next(line.split()[1] for line in cpp_out.splitlines()
                        if line.startswith("ns_per_command")))
    py_ns = python_ns_per_command(cmds, args.reps)
    print(f"machine        : {platform.machine()} / {platform.system()} {platform.release()}")
    print(f"python         : {platform.python_version()}")
    print(f"workload       : {len(cmds):,} commands (baseline.yaml, {args.steps} steps, "
          f"seed {args.seed}); outputs verified identical")
    print(f"python book    : {py_ns:,.0f} ns/command (median of {args.reps})")
    print(f"c++ book       : {cpp_ns:,.0f} ns/command (median of {args.reps})")
    print(f"ratio          : {py_ns / cpp_ns:,.1f}x (matching component only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
