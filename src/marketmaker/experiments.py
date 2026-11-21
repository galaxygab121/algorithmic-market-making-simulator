"""Experiments: artifact export, multi-seed / multi-regime suites, paired comparison.

Artifacts (``export_artifacts``) are written to ``<output>/<UTC timestamp>_seed<seed>/``:

* ``manifest.json``  — config (full), seed, config fingerprint, data source, versions,
  disclaimer, created-at timestamp
* ``metrics.json`` / ``comparison.csv`` — metrics per strategy
* ``<strategy>_timeseries.csv``, ``<strategy>_fills.csv``, ``<strategy>_orders.csv``,
  ``<strategy>_quotes.csv``, ``<strategy>_depth.csv``
* ``market_fundamental.csv`` — latent price path (analysis only)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import DISCLAIMER, __version__
from .config import RegimeSegment, SimulationConfig, load_config
from .simulation import ComparisonResult, run_comparison


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def export_artifacts(result: ComparisonResult, output: str | Path,
                     timestamp: str | None = None) -> Path:
    stamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(output) / f"{stamp}_seed{result.seed}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "disclaimer": DISCLAIMER,
        "simulated": True,
        "created_at_utc": stamp,
        "package_version": __version__,
        "python": platform.python_version(),
        "seed": result.seed,
        "config_fingerprint": result.config.fingerprint(),
        "data_source": result.stream.source,
        "strategies": list(result.runs),
        "config": result.config.model_dump(mode="json"),
    }
    (out / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    metrics = {name: r.metrics for name, r in result.runs.items()}
    (out / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    result.metrics_table().to_csv(out / "comparison.csv")
    for name, r in result.runs.items():
        r.timeseries.to_csv(out / f"{name}_timeseries.csv", index=False)
        r.fills.to_csv(out / f"{name}_fills.csv", index=False)
        r.orders.to_csv(out / f"{name}_orders.csv", index=False)
        r.quotes.to_csv(out / f"{name}_quotes.csv", index=False)
        r.depth.to_csv(out / f"{name}_depth.csv", index=False)
    pd.DataFrame({
        "time_ms": result.stream.step_times,
        "fundamental": result.stream.fundamental,
        "regime": result.stream.regimes,
    }).to_csv(out / "market_fundamental.csv", index=False)
    return out


REGIME_SUITE = ("baseline", "low_liquidity", "volatility_spike", "trend")


def with_regime(cfg: SimulationConfig, regime: str) -> SimulationConfig:
    """Return a copy of ``cfg`` whose whole run is in a single named regime."""
    data = cfg.model_dump(mode="python")
    data["market"]["regimes"] = [RegimeSegment(regime=regime).model_dump()]
    data["name"] = f"{cfg.name}:{regime}"
    return SimulationConfig.model_validate(data)


def run_suite(cfg: SimulationConfig, seeds: list[int],
              regimes: tuple[str, ...] = REGIME_SUITE) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in regimes:
        rcfg = with_regime(cfg, regime)
        for seed in seeds:
            res = run_comparison(rcfg, seed=seed)
            for name, run in res.runs.items():
                rows.append({"regime": regime, "seed": seed, "strategy": name, **run.metrics})
    return pd.DataFrame(rows)


def paired_comparison(suite: pd.DataFrame, metric: str = "final_equity_liquidated"
                      ) -> pd.DataFrame:
    """Per regime: mean of each strategy and a paired t-test (same seeds) of
    inventory_aware - baseline. With few seeds the p-value is only indicative."""
    out = []
    for regime, grp in suite.groupby("regime", sort=False):
        piv = grp.pivot(index="seed", columns="strategy", values=metric).dropna()
        diff = piv["inventory_aware"] - piv["baseline"]
        if len(diff) >= 2 and diff.std(ddof=1) > 0:
            p_value: float | None = float(
                stats.ttest_rel(piv["inventory_aware"], piv["baseline"]).pvalue)
        else:
            p_value = None
        out.append({
            "regime": regime, "n_seeds": len(piv),
            "baseline_mean": piv["baseline"].mean(),
            "inventory_aware_mean": piv["inventory_aware"].mean(),
            "mean_diff": diff.mean(), "paired_t_pvalue": p_value,
        })
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mmsim-experiments",
                                description="Multi-seed, multi-regime simulated comparison.")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--regimes", nargs="+", default=list(REGIME_SUITE),
                   choices=list(REGIME_SUITE))
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output", default=None, help="CSV path for the per-run table")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.steps:
        cfg = cfg.with_overrides(steps=args.steps)
    suite = run_suite(cfg, args.seeds, tuple(args.regimes))
    summary = paired_comparison(suite)
    print(DISCLAIMER)
    cols = ["regime", "strategy", "final_equity_liquidated", "max_drawdown", "inventory_std",
            "adverse_selection_ticks_per_unit", "kill_switch_tripped"]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(suite.groupby(["regime", "strategy"], sort=False)[cols[2:]].mean(numeric_only=True))
        print("\nPaired comparison (final_equity_liquidated):")
        print(summary.to_string(index=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        suite.to_csv(args.output, index=False)
        print(f"\nPer-run table written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
