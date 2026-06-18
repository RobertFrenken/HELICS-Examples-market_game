"""Belief-driven controllers for market-game agents."""

from __future__ import annotations

from dataclasses import dataclass

from ..actions import TargetLoad
from ..features import (
    distance_to_nearest_pricing_threshold,
    estimate_others_average_load,
    invert_price_to_average_load,
    recent_mean,
    recent_volatility,
)
from ..interfaces import AgentState, BaseController
from ..percepts import MarketPercept
from ..state import InferenceBeliefState


@dataclass
class LegalInferenceController(BaseController[TargetLoad]):
    """Infer aggregate pressure from legal delayed prices and own history."""

    house_count: int = 3

    def decide(self, percept: MarketPercept, state: AgentState) -> TargetLoad:
        belief = self._belief(state)
        previous_prices = percept.price_history[:-1]
        reference = recent_mean(previous_prices, fallback=percept.price, window=6)
        volatility = recent_volatility(previous_prices, window=6)
        inferred_average = None
        tier_distance = 999.0

        if percept.hour > 0 and belief.own_load_history:
            inverse = invert_price_to_average_load(percept.price)
            inferred_average = inverse.estimate
            tier_distance = distance_to_nearest_pricing_threshold(inferred_average)
            others_average = estimate_others_average_load(
                inferred_average,
                belief.own_load_history[-1],
                self.house_count,
            )
            crowd_delta = others_average - percept.demand[percept.hour - 1]
            belief.crowd_battery = min(
                percept.config.battery_capacity,
                max(0.0, belief.crowd_battery + crowd_delta),
            )

        future_window = [
            percept.demand[(percept.hour + offset) % len(percept.demand)]
            for offset in range(1, 4)
        ]
        future_pressure = sum(future_window) / len(future_window)
        remaining_capacity = percept.config.battery_capacity - percept.battery_charge

        cheap = percept.price < 0.95 * reference
        expensive = percept.price > 1.05 * reference
        upcoming_peak = future_pressure >= percept.base_demand + 2.0
        crowd_depleted = belief.crowd_battery < 3.0
        near_tier_boundary = tier_distance < 0.4

        if expensive and percept.battery_charge > 0.0:
            reserve = 4.0 if upcoming_peak and percept.hour < 20 else 0.0
            discharge_amount = min(
                percept.config.max_discharge,
                max(0.0, percept.battery_charge - reserve),
            )
            return self._remember(belief, percept.base_demand - discharge_amount)

        if cheap and remaining_capacity > 0.0 and not near_tier_boundary:
            charge_amount = min(percept.config.max_charge, remaining_capacity)
            if volatility > 0.20 or crowd_depleted:
                charge_amount *= 0.5
            return self._remember(belief, percept.base_demand + charge_amount)

        if percept.price <= 0.12 and remaining_capacity > 0.0:
            charge_amount = min(percept.config.max_charge, remaining_capacity)
            return self._remember(belief, percept.base_demand + charge_amount)

        if percept.hour >= 21 and percept.battery_charge > 0.0:
            discharge_amount = min(percept.config.max_discharge, percept.battery_charge)
            return self._remember(belief, percept.base_demand - discharge_amount)

        if inferred_average is not None and inferred_average >= 9.0 and percept.battery_charge > 2.0:
            discharge_amount = min(percept.config.max_discharge, percept.battery_charge - 2.0)
            return self._remember(belief, percept.base_demand - discharge_amount)

        return self._remember(belief, percept.base_demand)

    def _remember(self, belief: InferenceBeliefState, load: float) -> TargetLoad:
        belief.own_load_history.append(load)
        return TargetLoad(load)

    def _belief(self, state: AgentState) -> InferenceBeliefState:
        if not isinstance(state, InferenceBeliefState):
            raise TypeError("LegalInferenceController requires InferenceBeliefState")
        return state
