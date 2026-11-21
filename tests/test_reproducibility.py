from __future__ import annotations

import json

import numpy as np
import pandas as pd

from marketmaker.config import SimulationConfig
from marketmaker.marketdata import generate_market_data
from marketmaker.simulation import run_comparison


def _canonical_metrics(m: dict[str, object]) -> str:
    return json.dumps(m, sort_keys=True, default=str)


def test_same_seed_same_events(small_cfg: SimulationConfig) -> None:
    a = generate_market_data(small_cfg)
    b = generate_market_data(small_cfg)
    assert a.events == b.events
    np.testing.assert_array_equal(a.fundamental, b.fundamental)


def test_identical_seed_identical_trade_log_and_metrics(small_cfg: SimulationConfig) -> None:
    r1 = run_comparison(small_cfg)
    r2 = run_comparison(small_cfg)
    for name in r1.runs:
        pd.testing.assert_frame_equal(r1.runs[name].fills, r2.runs[name].fills)
        pd.testing.assert_frame_equal(r1.runs[name].orders, r2.runs[name].orders)
        pd.testing.assert_frame_equal(r1.runs[name].timeseries, r2.runs[name].timeseries)
        assert _canonical_metrics(r1.runs[name].metrics) == _canonical_metrics(
            r2.runs[name].metrics)


def test_different_seed_changes_observations(small_cfg: SimulationConfig) -> None:
    r1 = run_comparison(small_cfg, seed=1)
    r2 = run_comparison(small_cfg, seed=2)
    assert not np.array_equal(r1.stream.fundamental, r2.stream.fundamental)
    assert not r1.runs["baseline"].fills.equals(r2.runs["baseline"].fills)


def test_both_strategies_see_identical_market_stream(small_cfg: SimulationConfig) -> None:
    """Fairness: the exogenous events are generated once and replayed for each strategy;
    every strategy processes every exogenous event."""
    res = run_comparison(small_cfg)
    n_exo = len(res.stream.events)
    for run in res.runs.values():
        assert run.events_processed >= n_exo
        np.testing.assert_array_equal(run.timeseries["fundamental"].to_numpy()[:-1],
                                      res.stream.fundamental[1:])


def test_price_path_independent_of_lp_parameters(small_cfg: SimulationConfig) -> None:
    a = generate_market_data(small_cfg)
    b = generate_market_data(small_cfg.with_overrides(**{"market.lp_arrival_rate": 9.0}))
    np.testing.assert_array_equal(a.fundamental, b.fundamental)
