"""Baseline controllers for market-game agents."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..actions import BatteryDelta, FollowDemand, TargetLoad
from ..interfaces import AgentState, BaseController
from ..percepts import MarketPercept


@dataclass(frozen=True)
class FollowDemandController(BaseController[FollowDemand]):
    """Passive controller that submits the current base demand."""

    def decide(self, percept: MarketPercept, state: AgentState) -> FollowDemand:
        del percept, state
        return FollowDemand()


@dataclass(frozen=True)
class InvalidDemandController(BaseController[TargetLoad]):
    """Stress controller that deliberately submits illegal market loads."""

    load_offset: float = 100.0

    def decide(self, percept: MarketPercept, state: AgentState) -> TargetLoad:
        del state
        return TargetLoad(percept.base_demand + self.load_offset)


@dataclass(frozen=True)
class FlattenDemandController(BaseController[BatteryDelta]):
    """Price-blind controller that uses the battery to flatten own demand."""

    def decide(self, percept: MarketPercept, state: AgentState) -> BatteryDelta:
        del state
        target_demand = sum(percept.demand) / len(percept.demand)
        desired_change = target_demand - percept.base_demand

        if desired_change > 0.0:
            return BatteryDelta(
                min(
                    desired_change,
                    percept.config.max_charge,
                    percept.config.battery_capacity - percept.battery_charge,
                )
            )

        return BatteryDelta(
            -min(
                abs(desired_change),
                percept.config.max_discharge,
                percept.battery_charge,
            )
        )


@dataclass
class FullCycleController(BaseController[BatteryDelta]):
    """Mechanical controller that cycles the battery between full and empty."""

    charging: bool = True

    def reset(self) -> None:
        self.charging = True

    def decide(self, percept: MarketPercept, state: AgentState) -> BatteryDelta:
        del state
        if self.charging and percept.battery_charge >= percept.config.battery_capacity:
            self.charging = False
        elif (not self.charging) and percept.battery_charge <= 0.0:
            self.charging = True

        if self.charging:
            return BatteryDelta(
                min(
                    percept.config.max_charge,
                    percept.config.battery_capacity - percept.battery_charge,
                )
            )

        return BatteryDelta(-min(percept.config.max_discharge, percept.battery_charge))


@dataclass(frozen=True)
class OscillatingController(BaseController[BatteryDelta]):
    """Price-blind controller with sinusoidal charge/discharge swings."""

    period: int = 4
    phase: int = 0

    def decide(self, percept: MarketPercept, state: AgentState) -> BatteryDelta:
        del state
        wave = math.sin(2.0 * math.pi * (percept.hour + self.phase) / self.period)
        if wave >= 0.0:
            return BatteryDelta(
                min(
                    percept.config.max_charge,
                    percept.config.battery_capacity - percept.battery_charge,
                )
            )
        return BatteryDelta(-min(percept.config.max_discharge, percept.battery_charge))


@dataclass(frozen=True)
class VolatilitySeekingController(BaseController[BatteryDelta]):
    """Controller that tends to amplify recent price movement."""

    def decide(self, percept: MarketPercept, state: AgentState) -> BatteryDelta:
        del state
        previous_prices = percept.price_history[:-1]
        trend = (
            previous_prices[-1] - previous_prices[-2]
            if len(previous_prices) >= 2
            else 0.0
        )
        remaining_capacity = percept.config.battery_capacity - percept.battery_charge

        if percept.price < 0.49 and trend >= 0.0 and remaining_capacity > 0.0:
            return BatteryDelta(min(percept.config.max_charge, remaining_capacity))

        if percept.price >= 0.19 and percept.battery_charge > 0.0:
            return BatteryDelta(-min(percept.config.max_discharge, percept.battery_charge))

        if percept.hour >= 21 and percept.battery_charge > 0.0:
            return BatteryDelta(-min(percept.config.max_discharge, percept.battery_charge))

        return BatteryDelta(0.0)
