"""Threshold controllers for market-game agents."""

from __future__ import annotations

from dataclasses import dataclass

from python.market_game_downstream.core import DEFAULT_CONFIG
from ..contexts import MarketPercept
from ..interfaces import AgentState
from ..market_actions import BatteryDelta


BATTERY_MAX_CHARGE = DEFAULT_CONFIG.max_charge
BATTERY_MAX_DISCHARGE = DEFAULT_CONFIG.max_discharge


@dataclass(frozen=True)
class PriceAwareController:
    """Threshold controller with simple price bands and time-of-day reserves."""

    def reserve_target(self, hour: int) -> float:
        if hour < 12:
            return 4.0
        if hour < 18:
            return 8.0
        if hour < 21:
            return 3.0
        return 0.0

    def decide(self, percept: MarketPercept, state: AgentState) -> BatteryDelta:
        del state
        remaining_capacity = percept.config.battery_capacity - percept.battery_charge
        reserve = self.reserve_target(percept.hour)
        available_discharge = max(0.0, percept.battery_charge - reserve)

        if percept.price <= 0.12:
            return BatteryDelta(min(BATTERY_MAX_CHARGE, remaining_capacity))

        if percept.price <= 0.19 and percept.battery_charge < reserve:
            return BatteryDelta(
                min(BATTERY_MAX_CHARGE, remaining_capacity, reserve - percept.battery_charge)
            )

        if percept.price >= 0.49:
            return BatteryDelta(-min(BATTERY_MAX_DISCHARGE, percept.battery_charge))

        if percept.price >= 0.25 and available_discharge > 0.0:
            return BatteryDelta(-min(BATTERY_MAX_DISCHARGE, available_discharge))

        if percept.hour >= 21 and percept.battery_charge > 0.0:
            return BatteryDelta(-min(BATTERY_MAX_DISCHARGE, percept.battery_charge))

        return BatteryDelta(0.0)
