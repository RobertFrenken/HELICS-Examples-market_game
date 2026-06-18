"""Baseline policies for the pure market-game simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from python.market_game_downstream.core import DEFAULT_CONFIG
from .compose import MarketAgent
from .controllers import FollowDemandController, PriceAwareController
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
    """Passive baseline that submits the base demand profile exactly."""

    name: str = "FollowDemandHouse"
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = MarketAgent(
            name=self.name,
            controller=FollowDemandController(),
        )

    def reset(self) -> None:
        self._agent.reset()

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )


@dataclass
class InvalidDemandPolicy:
    """Stress opponent that deliberately submits illegal market loads."""

    name: str = "InvalidDemandHouse"
    load_offset: float = 100.0

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
        return demand[hour] + self.load_offset


@dataclass
class FlattenDemandPolicy:
    """Price-blind baseline that uses the battery to flatten own demand."""

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
    """Mechanical baseline that cycles the battery between full and empty."""

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
    """Threshold policy with simple price bands and time-of-day reserves."""

    name: str = "PriceAwareHouse"
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = MarketAgent(
            name=self.name,
            controller=PriceAwareController(),
        )

    def reset(self) -> None:
        self._agent.reset()

    def reserve_target(self, hour: int) -> float:
        return PriceAwareController().reserve_target(hour)

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )


@dataclass
class RollingPricePolicy:
    """Legal adaptive policy that compares price to a rolling recent mean."""

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
    """Legal-observation policy that infers aggregate pressure from prices."""

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


@dataclass
class NoisyThresholdPolicy:
    """Seeded threshold opponent with repeatable per-hour jitter."""

    name: str = "NoisyThresholdHouse"
    seed: int = 1
    reserve: float = 4.0
    noise_scale: float = 0.04
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

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
        cheap_threshold = 0.14 + self._rng.uniform(-self.noise_scale, self.noise_scale)
        expensive_threshold = 0.34 + self._rng.uniform(-self.noise_scale, self.noise_scale)

        if price <= cheap_threshold and remaining_capacity > 0.0:
            return base_demand + min(BATTERY_MAX_CHARGE, remaining_capacity)

        if price >= expensive_threshold and battery_charge > self.reserve:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge - self.reserve)
            return base_demand - discharge_amount

        if hour >= 21 and battery_charge > 0.0:
            return base_demand - min(BATTERY_MAX_DISCHARGE, battery_charge)

        return base_demand


@dataclass
class OscillatingPolicy:
    """Price-blind opponent with sinusoidal charge/discharge swings."""

    name: str = "OscillatingHouse"
    period: int = 4
    phase: int = 0

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
        wave = math.sin(2.0 * math.pi * (hour + self.phase) / self.period)
        if wave >= 0.0:
            charge_amount = min(BATTERY_MAX_CHARGE, BATTERY_CAPACITY - battery_charge)
            return base_demand + charge_amount
        discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
        return base_demand - discharge_amount


@dataclass
class VolatilitySeekingPolicy:
    """Chaotic opponent that tends to amplify price movement."""

    name: str = "VolatilitySeekingHouse"

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
        trend = (
            previous_prices[-1] - previous_prices[-2]
            if len(previous_prices) >= 2
            else 0.0
        )
        remaining_capacity = BATTERY_CAPACITY - battery_charge

        if price < 0.49 and trend >= 0.0 and remaining_capacity > 0.0:
            return base_demand + min(BATTERY_MAX_CHARGE, remaining_capacity)

        if price >= 0.19 and battery_charge > 0.0:
            return base_demand - min(BATTERY_MAX_DISCHARGE, battery_charge)

        if hour >= 21 and battery_charge > 0.0:
            return base_demand - min(BATTERY_MAX_DISCHARGE, battery_charge)

        return base_demand


POLICY_TYPES = {
    "FlattenDemandPolicy": FlattenDemandPolicy,
    "FollowDemandPolicy": FollowDemandPolicy,
    "FullCyclePolicy": FullCyclePolicy,
    "InvalidDemandPolicy": InvalidDemandPolicy,
    "LegalInferencePolicy": LegalInferencePolicy,
    "NoisyThresholdPolicy": NoisyThresholdPolicy,
    "OscillatingPolicy": OscillatingPolicy,
    "PriceAwarePolicy": PriceAwarePolicy,
    "RollingPricePolicy": RollingPricePolicy,
    "VolatilitySeekingPolicy": VolatilitySeekingPolicy,
}
