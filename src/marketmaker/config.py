"""Validated simulation configuration (Pydantic v2), loadable from YAML.

Units
-----
* Time: integer milliseconds of *simulated* time.
* Prices: ``initial_price`` and ``tick_size`` in currency; everything inside the engine is
  integer ticks. ``volatility`` is the relative standard deviation per sqrt(second)
  (e.g. 0.0002 → ~2 ticks/sqrt(s) at price 100.00 with a 0.01 tick).
* Fees: basis points of notional. Negative maker fee = rebate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

RegimeName = Literal["baseline", "low_liquidity", "volatility_spike", "trend"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegimeSegment(_Strict):
    """A named market regime active over a fraction of the run. Later segments override
    earlier ones where they overlap. Optional fields override the preset's values."""

    regime: RegimeName = "baseline"
    start_frac: float = Field(0.0, ge=0.0, le=1.0)
    end_frac: float = Field(1.0, ge=0.0, le=1.0)
    volatility_multiplier: float | None = Field(None, gt=0)
    liquidity_multiplier: float | None = Field(None, gt=0)
    flow_multiplier: float | None = Field(None, ge=0)
    cancel_multiplier: float | None = Field(None, gt=0)
    drift_per_sec: float | None = None
    flow_imbalance: float | None = Field(None, ge=-0.5, le=0.5)

    @model_validator(mode="after")
    def _ordered(self) -> RegimeSegment:
        if self.end_frac <= self.start_frac:
            raise ValueError("end_frac must be greater than start_frac")
        return self


class MarketConfig(_Strict):
    initial_price: float = Field(100.0, gt=0)
    process: Literal["gbm", "ou"] = "gbm"
    volatility: float = Field(0.0002, ge=0)
    drift_per_sec: float = 0.0
    ou_kappa: float = Field(0.02, ge=0, description="OU mean-reversion speed per second")
    # Background liquidity providers (LP)
    lp_arrival_rate: float = Field(3.0, ge=0, description="expected LP orders per side/step")
    lp_max_levels: int = Field(8, ge=1)
    lp_level_decay: float = Field(0.35, gt=0, le=1, description="geometric p for distance")
    lp_max_size: int = Field(10, ge=1)
    lp_cancel_prob: float = Field(0.04, ge=0, le=1, description="per order per step")
    initial_depth_per_level: int = Field(10, ge=1)
    # Takers
    market_order_rate: float = Field(0.5, ge=0, description="expected market orders/step")
    market_order_max_size: int = Field(8, ge=1)
    informed_fraction: float = Field(0.15, ge=0, le=1)
    informed_horizon_steps: int = Field(10, ge=1)
    flow_imbalance: float = Field(0.0, ge=-0.5, le=0.5, description="P(buy) - 0.5")
    regimes: list[RegimeSegment] = Field(default_factory=lambda: [RegimeSegment()])
    price_csv: str | None = Field(
        None, description="optional CSV of mid prices that replaces the synthetic path"
    )


class FeeConfig(_Strict):
    maker_fee_bps: float = -0.2
    taker_fee_bps: float = 0.5


class ExchangeConfig(_Strict):
    tick_size: float = Field(0.01, gt=0)
    contract_multiplier: float = Field(10.0, gt=0)
    fees: FeeConfig = Field(default_factory=FeeConfig)
    order_latency_ms: int = Field(20, ge=0)
    cancel_latency_ms: int = Field(20, ge=0)
    self_trade_prevention: bool = True

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.contract_multiplier


class RiskConfig(_Strict):
    enabled: bool = True
    position_limit: int = Field(100, ge=1)
    max_order_qty: int = Field(20, ge=1)
    max_notional: float = Field(100_000.0, gt=0)
    max_drawdown: float = Field(100.0, gt=0, description="currency, peak-to-trough")
    flatten_on_kill: bool = Field(
        False, description="after the kill switch cancels quotes, close inventory with a "
        "taker market order (bypasses pre-trade checks)"
    )


class QuotingConfig(_Strict):
    quote_size: int = Field(5, ge=1)
    max_inventory: int = Field(50, ge=1, description="strategy-level soft cap")
    decision_interval_ms: int = Field(100, ge=1)
    min_quote_lifetime_ms: int = Field(200, ge=0)
    max_messages_per_sec: int = Field(20, ge=1)
    requote_threshold_ticks: int = Field(1, ge=1)


class BaselineStrategyConfig(QuotingConfig):
    half_spread_ticks: int = Field(2, ge=1)


class InventoryStrategyConfig(QuotingConfig):
    gamma: float = Field(0.01, gt=0, description="risk aversion per unit inventory")
    horizon_sec: float = Field(10.0, gt=0, description="fixed risk horizon tau")
    k: float = Field(1.5, gt=0, description="order-arrival decay (1/ticks)")
    min_half_spread_ticks: int = Field(1, ge=1)
    max_spread_ticks: int = Field(20, ge=2)
    sigma_prior_ticks: float | None = Field(None, gt=0, description="ticks/sqrt(s)")
    vol_halflife_sec: float = Field(30.0, gt=0)


class StrategiesConfig(_Strict):
    baseline: BaselineStrategyConfig = Field(default_factory=BaselineStrategyConfig)
    inventory_aware: InventoryStrategyConfig = Field(default_factory=InventoryStrategyConfig)


class SimulationConfig(_Strict):
    name: str = "baseline"
    seed: int = 42
    steps: int = Field(3000, ge=10)
    step_ms: int = Field(100, ge=1)
    market: MarketConfig = Field(default_factory=MarketConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    liquidation_penalty_ticks: float = Field(
        1.0, ge=0, description="end-of-run liquidation cost per unit vs mid, plus taker fee"
    )
    markout_horizon_ms: int = Field(1000, ge=1)
    sharpe_sample_ms: int = Field(1000, ge=1)
    depth_snapshot_every: int = Field(50, ge=1, description="samples between depth snapshots")

    @property
    def duration_ms(self) -> int:
        return self.steps * self.step_ms

    def fingerprint(self) -> str:
        """Stable hash of the full configuration (including seed)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def with_overrides(self, **updates: Any) -> SimulationConfig:
        """Return a validated copy with dotted-path overrides, e.g.
        ``cfg.with_overrides(**{"market.volatility": 0.001, "seed": 7})``."""
        data = self.model_dump(mode="python")
        for dotted, value in updates.items():
            node = data
            *parents, leaf = dotted.split(".")
            for p in parents:
                node = node[p]
            if leaf not in node:
                raise KeyError(f"unknown config key: {dotted}")
            node[leaf] = value
        return SimulationConfig.model_validate(data)


def load_config(path: str | Path) -> SimulationConfig:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return SimulationConfig.model_validate(data)


def dump_config_yaml(cfg: SimulationConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
