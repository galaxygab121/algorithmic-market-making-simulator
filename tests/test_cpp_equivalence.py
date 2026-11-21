"""Python ↔ C++ matching-engine equivalence. Skipped unless the optional C++ engine has
been built (cmake -S cpp -B cpp/build && cmake --build cpp/build)."""

from __future__ import annotations

from pathlib import Path

import pytest

from marketmaker.config import SimulationConfig
from marketmaker.cpp_bridge import build_workload, find_engine, python_reference, run_engine

ENGINE = find_engine()
pytestmark = pytest.mark.skipif(ENGINE is None, reason="C++ engine not built")


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("regime", ["baseline", "low_liquidity", "volatility_spike"])
def test_identical_outputs(tmp_path: Path, small_cfg: SimulationConfig, seed: int,
                           regime: str) -> None:
    from marketmaker.experiments import with_regime

    cfg = with_regime(small_cfg, regime)
    cmds = build_workload(cfg, seed)
    py = python_reference(cmds)
    assert ENGINE is not None
    cpp = run_engine(cmds, ENGINE, tmp_path)
    assert py.count("\nT ") > 100 and "post-only order would cross" in py
    assert "duplicate order id" in py and "\nS " in py  # all paths exercised
    assert cpp == py
