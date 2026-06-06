"""Baseline policies for the pure market-game simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFAULT_CONFIG
from .features import (
    distance_to_nearest_pricing_threshold,
    estimate_others_average_load,
    invert_price_to_average_load,
    recent_mean,
    recent_volatility,
)

BATTERY_CAPACITY = DEFAULT_CONFIG.battery_capacity
BATTERY_MAX_CHARGE = DEFAULT_CONFIG.max_charge
BATTERY_MAX_DISCHARGE = DEFAULT_CONFIG.max_discharge


@dataclass
class FollowDemandPolicy:
    name: str = "FollowDemandHouse"

    def reset(self) -> None:
        pass

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        del price, battery_charge, price_history
        return demand[hour]


@dataclass
class FlattenDemandPolicy:
    name: str = "FlattenDemandHouse"

    def reset(self) -> None:
        pass

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        del price, price_history
        base_demand = demand[hour]
        target_demand = sum(demand) / len(demand)
        desired_change = target_demand - base_demand

        if desired_change > 0.0:
            charge_amount = min(desired_change, BATTERY_MAX_CHARGE, BATTERY_CAPACITY - battery_charge)
            return base_demand + charge_amount

        discharge_amount = min(abs(desired_change), BATTERY_MAX_DISCHARGE, battery_charge)
        return base_demand - discharge_amount


@dataclass
class FullCyclePolicy:
    name: str = "FullCycleHouse"
    charging: bool = True

    def reset(self) -> None:
        self.charging = True

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        del price, price_history

        if self.charging and battery_charge >= BATTERY_CAPACITY:
            self.charging = False
        elif (not self.charging) and battery_charge <= 0.0:
            self.charging = True

        if self.charging:
            charge_amount = min(BATTERY_MAX_CHARGE, BATTERY_CAPACITY - battery_charge)
            return demand[hour] + charge_amount

        discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
        return demand[hour] - discharge_amount


@dataclass
class PriceAwarePolicy:
    name: str = "PriceAwareHouse"

    def reset(self) -> None:
        pass

    def reserve_target(self, hour: int) -> float:
        if hour < 12:
            return 4.0
        if hour < 18:
            return 8.0
        if hour < 21:
            return 3.0
        return 0.0

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        del price_history

        base_demand = demand[hour]
        remaining_capacity = BATTERY_CAPACITY - battery_charge
        reserve = self.reserve_target(hour)
        available_discharge = max(0.0, battery_charge - reserve)

        if price <= 0.12:
            charge_amount = min(BATTERY_MAX_CHARGE, remaining_capacity)
            return base_demand + charge_amount

        if price <= 0.19 and battery_charge < reserve:
            charge_amount = min(BATTERY_MAX_CHARGE, remaining_capacity, reserve - battery_charge)
            return base_demand + charge_amount

        if price >= 0.49:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
            return base_demand - discharge_amount

        if price >= 0.25 and available_discharge > 0.0:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, available_discharge)
            return base_demand - discharge_amount

        if hour >= 21 and battery_charge > 0.0:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
            return base_demand - discharge_amount

        return base_demand


@dataclass
class RollingPricePolicy:
    name: str = "RollingPriceHouse"
    window: int = 6
    cheap_ratio: float = 0.94
    expensive_ratio: float = 1.08
    reserve: float = 4.0

    def reset(self) -> None:
        pass

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        base_demand = demand[hour]
        previous_prices = price_history[:-1]
        reference = recent_mean(previous_prices, fallback=price, window=self.window)
        volatility = recent_volatility(previous_prices, window=self.window)
        remaining_capacity = BATTERY_CAPACITY - battery_charge

        cheap = price < self.cheap_ratio * reference
        expensive = price > self.expensive_ratio * reference

        if cheap and remaining_capacity > 0.0:
            charge_amount = min(BATTERY_MAX_CHARGE, remaining_capacity)
            if volatility > 0.15:
                charge_amount *= 0.5
            return base_demand + charge_amount

        if expensive and battery_charge > self.reserve:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge - self.reserve)
            return base_demand - discharge_amount

        if hour >= 21 and battery_charge > 0.0:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
            return base_demand - discharge_amount

        return base_demand


@dataclass
class LegalInferencePolicy:
    name: str = "LegalInferenceHouse"
    house_count: int = 3
    own_load_history: list[float] = field(default_factory=list)
    crowd_battery_belief: float = 0.0

    def reset(self) -> None:
        self.own_load_history.clear()
        self.crowd_battery_belief = 0.0

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        base_demand = demand[hour]
        previous_prices = price_history[:-1]
        reference = recent_mean(previous_prices, fallback=price, window=6)
        volatility = recent_volatility(previous_prices, window=6)
        inferred_average = None
        tier_distance = 999.0

        if hour > 0 and self.own_load_history:
            inverse = invert_price_to_average_load(price)
            inferred_average = inverse.estimate
            tier_distance = distance_to_nearest_pricing_threshold(inferred_average)
            others_average = estimate_others_average_load(
                inferred_average,
                self.own_load_history[-1],
                self.house_count,
            )
            crowd_delta = others_average - demand[hour - 1]
            self.crowd_battery_belief = min(
                BATTERY_CAPACITY,
                max(0.0, self.crowd_battery_belief + crowd_delta),
            )

        future_window = [demand[(hour + offset) % len(demand)] for offset in range(1, 4)]
        future_pressure = sum(future_window) / len(future_window)
        remaining_capacity = BATTERY_CAPACITY - battery_charge

        cheap = price < 0.95 * reference
        expensive = price > 1.05 * reference
        upcoming_peak = future_pressure >= base_demand + 2.0
        crowd_depleted = self.crowd_battery_belief < 3.0
        near_tier_boundary = tier_distance < 0.4

        if expensive and battery_charge > 0.0:
            reserve = 4.0 if upcoming_peak and hour < 20 else 0.0
            discharge_amount = min(BATTERY_MAX_DISCHARGE, max(0.0, battery_charge - reserve))
            proposed = base_demand - discharge_amount
            self.own_load_history.append(proposed)
            return proposed

        if cheap and remaining_capacity > 0.0 and not near_tier_boundary:
            charge_amount = min(BATTERY_MAX_CHARGE, remaining_capacity)
            if volatility > 0.20 or crowd_depleted:
                charge_amount *= 0.5
            proposed = base_demand + charge_amount
            self.own_load_history.append(proposed)
            return proposed

        if price <= 0.12 and remaining_capacity > 0.0:
            charge_amount = min(BATTERY_MAX_CHARGE, remaining_capacity)
            proposed = base_demand + charge_amount
            self.own_load_history.append(proposed)
            return proposed

        if hour >= 21 and battery_charge > 0.0:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
            proposed = base_demand - discharge_amount
            self.own_load_history.append(proposed)
            return proposed

        if inferred_average is not None and inferred_average >= 9.0 and battery_charge > 2.0:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge - 2.0)
            proposed = base_demand - discharge_amount
            self.own_load_history.append(proposed)
            return proposed

        self.own_load_history.append(base_demand)
        return base_demand
