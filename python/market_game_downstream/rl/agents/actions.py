"""Compatibility action mappers for market-game agents.

The game and submission ABI is always a finite market load. These adapters are
small conveniences that map strategy or learner actions onto that market-load
contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from python.market_game_downstream.core.config import DEFAULT_CONFIG, MarketGameConfig
from python.market_game_downstream.core.rules import (
    battery_delta_to_market_load,
    normalized_delta_to_market_load,
)
from .interfaces import ActionMapper
from .market_actions import BatteryPosture


@dataclass(frozen=True)
class DiscreteBatteryPostureActionSpace:
    """Three-action smoke/debug baseline for battery posture."""

    def market_load(
        self,
        action: object,
        base_demand: float,
        battery_charge: float,
        config: MarketGameConfig = DEFAULT_CONFIG,
    ) -> float:
        posture = BatteryPosture(action)
        if posture == BatteryPosture.DISCHARGE:
            delta = -config.max_discharge
        elif posture == BatteryPosture.NEUTRAL:
            delta = 0.0
        else:
            delta = config.max_charge
        return battery_delta_to_market_load(delta, base_demand, battery_charge, config)


@dataclass(frozen=True)
class IntegerBatteryDeltaActionSpace:
    """Discrete battery-delta baseline with integer kWh actions."""

    min_delta: int = -10
    max_delta: int = 5

    def market_load(
        self,
        action: object,
        base_demand: float,
        battery_charge: float,
        config: MarketGameConfig = DEFAULT_CONFIG,
    ) -> float:
        delta = int(action)
        if delta < self.min_delta or delta > self.max_delta:
            raise ValueError(
                f"invalid integer battery delta {action!r}; "
                f"expected {self.min_delta}..{self.max_delta}"
            )
        return battery_delta_to_market_load(delta, base_demand, battery_charge, config)


@dataclass(frozen=True)
class ContinuousNormalizedDeltaActionSpace:
    """Continuous baseline: action in [-1, 1] maps to the legal delta range."""

    def market_load(
        self,
        action: object,
        base_demand: float,
        battery_charge: float,
        config: MarketGameConfig = DEFAULT_CONFIG,
    ) -> float:
        return normalized_delta_to_market_load(
            float(action),
            base_demand,
            battery_charge,
            config,
        )


DEFAULT_ACTION_SPACE = DiscreteBatteryPostureActionSpace()
