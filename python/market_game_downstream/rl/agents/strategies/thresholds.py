"""Threshold strategies for composed market-game agents."""

from __future__ import annotations

from dataclasses import dataclass

from python.market_game_downstream.core import DEFAULT_CONFIG
from ..contexts import Observation
from ..interfaces import AgentState


@dataclass
class RollingThresholdStrategy:
    """Compare current price to recent-price features and emit a battery delta."""

    cheap_ratio: float = 0.94
    expensive_ratio: float = 1.08
    reserve: float = 4.0

    def decide(self, observation: Observation, state: AgentState) -> float:
        del state
        price = observation[2]
        battery_charge = observation[3]
        price_mean_recent = observation[7]
        price_volatility_recent = observation[9]
        remaining_capacity = DEFAULT_CONFIG.battery_capacity - battery_charge

        cheap = price < self.cheap_ratio * price_mean_recent
        expensive = price > self.expensive_ratio * price_mean_recent

        if cheap and remaining_capacity > 0.0:
            charge_amount = min(DEFAULT_CONFIG.max_charge, remaining_capacity)
            if price_volatility_recent > 0.15:
                charge_amount *= 0.5
            return charge_amount

        if expensive and battery_charge > self.reserve:
            return -min(DEFAULT_CONFIG.max_discharge, battery_charge - self.reserve)

        return 0.0

    def get_params(self) -> dict[str, object]:
        return {
            "cheap_ratio": self.cheap_ratio,
            "expensive_ratio": self.expensive_ratio,
            "reserve": self.reserve,
        }

    def set_params(self, **params: object) -> None:
        for name in params:
            if name not in {"cheap_ratio", "expensive_ratio", "reserve"}:
                raise ValueError(f"unknown RollingThresholdStrategy parameter {name!r}")
        if "cheap_ratio" in params:
            self.cheap_ratio = float(params["cheap_ratio"])
        if "expensive_ratio" in params:
            self.expensive_ratio = float(params["expensive_ratio"])
        if "reserve" in params:
            self.reserve = float(params["reserve"])
