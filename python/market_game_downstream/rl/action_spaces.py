"""RL action-space adapters for the market-game simulator.

The game and submission ABI is always a finite market load. These adapters are
only RL conveniences that map learner actions onto that market-load contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from python.market_game_downstream.core.config import DEFAULT_CONFIG, MarketGameConfig
from python.market_game_downstream.core.rules import (
    battery_delta_to_market_load,
    normalized_delta_to_market_load,
)


class ActionMapper(Protocol):
    """Map an RL action to a proposed market load for the current hour."""

    def market_load(
        self,
        action: object,
        base_demand: float,
        battery_charge: float,
        config: MarketGameConfig = DEFAULT_CONFIG,
    ) -> float:
        """Return the proposed market load for one learner action."""


class BatteryPosture(IntEnum):
    """Coarse starter action: max discharge, neutral, or max charge."""

    DISCHARGE = -1
    NEUTRAL = 0
    CHARGE = 1


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
