"""Project semantic market actions onto proposed market loads."""

from __future__ import annotations

from python.market_game_downstream.core.rules import battery_delta_to_market_load

from .primitives import BatteryDelta, BatteryPosture, MarketAction, MarketPercept, TargetLoad


def project_market_load(action: MarketAction, percept: MarketPercept) -> float:
    """Convert a semantic market action to the simulator load ABI."""
    if isinstance(action, TargetLoad):
        return float(action.load)
    if isinstance(action, BatteryDelta):
        return battery_delta_to_market_load(
            action.kwh,
            percept.base_demand,
            percept.battery_charge,
            percept.config,
        )
    if isinstance(action, BatteryPosture):
        return _posture_load(action, percept)
    return float(action)


def _posture_load(posture: BatteryPosture, percept: MarketPercept) -> float:
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
