from __future__ import annotations

from pathlib import Path

import pytest

from marketmaker.config import SimulationConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture
def small_cfg() -> SimulationConfig:
    """A short baseline run (60 simulated seconds) for fast integration tests."""
    return load_config(EXAMPLES / "baseline.yaml").with_overrides(steps=600)


@pytest.fixture
def stress_cfg() -> SimulationConfig:
    return load_config(EXAMPLES / "inventory_stress.yaml")
