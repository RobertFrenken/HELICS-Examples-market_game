"""Actuators that project strategy actions to market-facing loads."""

from __future__ import annotations

from dataclasses import dataclass

from python.market_game_downstream.core.rules import battery_delta_to_market_load
from .actions import (
    ContinuousNormalizedDeltaActionSpace,
    DiscreteBatteryPostureActionSpace,
    IntegerBatteryDeltaActionSpace,
)
from .contexts import MarketContext


@dataclass(frozen=True)
class DirectLoadActuator:
    """Interpret the internal action as the proposed market load."""

    def market_load(self, action: object, context: MarketContext) -> float:
        del context
        return float(action)


@dataclass(frozen=True)
class ActionMapperActuator:
    """Adapter from legacy action mappers to the composed-agent actuator API."""

    mapper: object

    def market_load(self, action: object, context: MarketContext) -> float:
        return float(
            self.mapper.market_load(
                action,
                context.base_demand,
                context.battery_charge,
                context.config,
            )
        )


@dataclass(frozen=True)
class BatteryDeltaActuator:
    """Interpret the internal action as a direct battery delta in kWh."""

    def market_load(self, action: object, context: MarketContext) -> float:
        return battery_delta_to_market_load(
            float(action),
            context.base_demand,
            context.battery_charge,
            context.config,
        )


BatteryPostureActuator = DiscreteBatteryPostureActionSpace
IntegerBatteryDeltaActuator = IntegerBatteryDeltaActionSpace
ContinuousNormalizedDeltaActuator = ContinuousNormalizedDeltaActionSpace
