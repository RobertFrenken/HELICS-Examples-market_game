"""Project semantic market actions onto proposed market loads."""

from __future__ import annotations

from dataclasses import dataclass

from python.market_game_downstream.core.rules import battery_delta_to_market_load
from .actions import (
    BatteryDelta,
    BatteryPosture,
    BatteryPostureAction,
    FollowDemand,
    MarketAction,
    TargetLoad,
)
from .percepts import MarketPercept


@dataclass(frozen=True)
class MarketActionProjector:
    """Convert domain-level market actions to the simulator load ABI."""

    def market_load(self, action: MarketAction, percept: MarketPercept) -> float:
        if isinstance(action, FollowDemand):
            return percept.base_demand
        if isinstance(action, TargetLoad):
            return float(action.load)
        if isinstance(action, BatteryDelta):
            return battery_delta_to_market_load(
                action.kwh,
                percept.base_demand,
                percept.battery_charge,
                percept.config,
            )
        if isinstance(action, BatteryPostureAction):
            return self._posture_load(action.posture, percept)
        return float(action)

    def _posture_load(self, posture: BatteryPosture, percept: MarketPercept) -> float:
        if posture == BatteryPosture.DISCHARGE:
            delta = -percept.config.max_discharge
        elif posture == BatteryPosture.NEUTRAL:
            delta = 0.0
        else:
            delta = percept.config.max_charge
        return battery_delta_to_market_load(
            delta,
            percept.base_demand,
            percept.battery_charge,
            percept.config,
        )
