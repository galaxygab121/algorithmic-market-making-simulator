"""One-command demo: runs the three example scenarios, exports artifacts, and writes
self-contained HTML charts plus a static PNG summary per scenario.

    uv run python scripts/run_demo.py            # -> runs/demo/...
    uv run python scripts/run_demo.py --readme   # also refresh docs/images/*.png

All output is SIMULATED.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marketmaker import DISCLAIMER  # noqa: E402
from marketmaker.config import load_config  # noqa: E402
from marketmaker.experiments import export_artifacts  # noqa: E402
from marketmaker.metrics import format_value  # noqa: E402
from marketmaker.plots import (  # noqa: E402
    comparison_figure,
    price_and_quotes_figure,
    save_html,
    save_summary_png,
)
from marketmaker.simulation import run_comparison  # noqa: E402

SCENARIOS = ("baseline", "volatile_market", "inventory_stress")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "runs" / "demo"))
    ap.add_argument("--readme", action="store_true", help="refresh docs/images PNGs")
    args = ap.parse_args()
    print(DISCLAIMER)
    for name in SCENARIOS:
        cfg = load_config(ROOT / "examples" / f"{name}.yaml")
        res = run_comparison(cfg)
        out = export_artifacts(res, Path(args.output) / name)
        save_html(comparison_figure(res), out / "comparison.html")
        for sname, run in res.runs.items():
            save_html(price_and_quotes_figure(run, cfg.exchange.tick_size),
                      out / f"{sname}_quotes.html")
        save_summary_png(res, out / "summary.png")
        if args.readme:
            save_summary_png(res, ROOT / "docs" / "images" / f"{name}_summary.png")
        print(f"\n== {name} (seed {res.seed}) -> {out}")
        for sname, run in res.runs.items():
            m = run.metrics
            print(f"  {sname:<16} liquidated equity={format_value(m['final_equity_liquidated'])}"
                  f"  trades={m['trade_count']}  maxDD={format_value(m['max_drawdown'])}"
                  f"  inv_std={format_value(m['inventory_std'])}"
                  f"  kill={'yes @ %.1fs' % (m['kill_switch_time_ms'] / 1000)
                           if m['kill_switch_tripped'] else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
