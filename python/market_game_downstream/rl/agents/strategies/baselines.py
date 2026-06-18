"""Baseline strategies for composed market-game agents."""

from __future__ import annotations

from dataclasses import dataclass

from python.market_game_downstream.core import DEFAULT_CONFIG
from ..contexts import Observation
from ..interfaces import AgentState

BATTERY_CAPACITY = DEFAULT_CONFIG.battery_capacity
BATTERY_MAX_CHARGE = DEFAULT_CONFIG.max_charge
BATTERY_MAX_DISCHARGE = DEFAULT_CONFIG.max_discharge


@dataclass(frozen=True)
class IdentityStrategy:
    """Return the first observation value as the internal action."""

    def decide(self, observation: Observation, state: AgentState) -> object:
        del state
        return observation[0]


@dataclass(frozen=True)
class FollowDemandStrategy:
    """Return the base demand from a local observation."""

    def decide(self, observation: Observation, state: AgentState) -> float:
        del state
        return float(observation[-1])


@dataclass
class FullCycleStrategy:
    """Mechanical strategy that emits battery deltas to cycle full and empty."""

    charging: bool = True

    def reset(self) -> None:
        self.charging = True

    def decide(self, observation: Observation, state: AgentState) -> float:
        del state
        battery_charge = observation[3]
        if self.charging and battery_charge >= BATTERY_CAPACITY:
            self.charging = False
        elif (not self.charging) and battery_charge <= 0.0:
            self.charging = True

        if self.charging:
            return min(BATTERY_MAX_CHARGE, BATTERY_CAPACITY - battery_charge)
        return -min(BATTERY_MAX_DISCHARGE, battery_charge)


@dataclass
class PriceAwareStrategy:
    """Threshold strategy that emits battery deltas from local observations."""

    def reserve_target(self, hour: int) -> float:
        if hour < 12:
            return 4.0
        if hour < 18:
            return 8.0
        if hour < 21:
            return 3.0
        return 0.0

    def decide(self, observation: Observation, state: AgentState) -> float:
        del state
        hour_sin, hour_cos, price, battery_charge, _base_demand = observation[:5]
        hour = _hour_from_angle(hour_sin, hour_cos)
        remaining_capacity = BATTERY_CAPACITY - battery_charge
        reserve = self.reserve_target(hour)
        available_discharge = max(0.0, battery_charge - reserve)

        if price <= 0.12:
            return min(BATTERY_MAX_CHARGE, remaining_capacity)
        if price <= 0.19 and battery_charge < reserve:
            return min(BATTERY_MAX_CHARGE, remaining_capacity, reserve - battery_charge)
        if price >= 0.49:
            return -min(BATTERY_MAX_DISCHARGE, battery_charge)
        if price >= 0.25 and available_discharge > 0.0:
            return -min(BATTERY_MAX_DISCHARGE, available_discharge)
        if hour >= 21 and battery_charge > 0.0:
            return -min(BATTERY_MAX_DISCHARGE, battery_charge)
        return 0.0


def _hour_from_angle(hour_sin: float, hour_cos: float) -> int:
    import math

    angle = math.atan2(hour_sin, hour_cos)
    if angle < 0.0:
        angle += 2.0 * math.pi
    return int(round(angle / (2.0 * math.pi) * DEFAULT_CONFIG.episode_hours)) % (
        DEFAULT_CONFIG.episode_hours
    )
