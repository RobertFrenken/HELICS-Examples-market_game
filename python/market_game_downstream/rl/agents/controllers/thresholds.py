"""Threshold controllers for market-game agents."""

from __future__ import annotations

from dataclasses import dataclass, field
import random

from python.market_game_downstream.core import DEFAULT_CONFIG
from ..primitives import BatteryDelta
from ..features import recent_mean, recent_volatility
from ..primitives import MarketPercept


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

    def decide(self, percept: MarketPercept, state: object) -> BatteryDelta:
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


@dataclass
class RollingThresholdController:
    """Compare current price to a rolling recent mean and emit battery deltas."""

    window: int = 6
    cheap_ratio: float = 0.94
    expensive_ratio: float = 1.08
    reserve: float = 4.0

    def decide(self, percept: MarketPercept, state: object) -> BatteryDelta:
        del state
        previous_prices = percept.price_history[:-1]
        reference = recent_mean(
            previous_prices,
            fallback=percept.price,
            window=self.window,
        )
        volatility = recent_volatility(previous_prices, window=self.window)
        remaining_capacity = percept.config.battery_capacity - percept.battery_charge

        cheap = percept.price < self.cheap_ratio * reference
        expensive = percept.price > self.expensive_ratio * reference

        if cheap and remaining_capacity > 0.0:
            charge_amount = min(percept.config.max_charge, remaining_capacity)
            if volatility > 0.15:
                charge_amount *= 0.5
            return BatteryDelta(charge_amount)

        if expensive and percept.battery_charge > self.reserve:
            return BatteryDelta(
                -min(percept.config.max_discharge, percept.battery_charge - self.reserve)
            )

        if percept.hour >= 21 and percept.battery_charge > 0.0:
            return BatteryDelta(-min(percept.config.max_discharge, percept.battery_charge))

        return BatteryDelta(0.0)

    def get_params(self) -> dict[str, object]:
        return {
            "window": self.window,
            "cheap_ratio": self.cheap_ratio,
            "expensive_ratio": self.expensive_ratio,
            "reserve": self.reserve,
        }

    def set_params(self, **params: object) -> None:
        allowed = {"window", "cheap_ratio", "expensive_ratio", "reserve"}
        for name in params:
            if name not in allowed:
                raise ValueError(f"unknown RollingThresholdController parameter {name!r}")
        if "window" in params:
            self.window = int(params["window"])
        if "cheap_ratio" in params:
            self.cheap_ratio = float(params["cheap_ratio"])
        if "expensive_ratio" in params:
            self.expensive_ratio = float(params["expensive_ratio"])
        if "reserve" in params:
            self.reserve = float(params["reserve"])


@dataclass
class NoisyThresholdController:
    """Seeded threshold controller with repeatable per-hour jitter."""

    seed: int = 1
    reserve: float = 4.0
    noise_scale: float = 0.04
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

    def decide(self, percept: MarketPercept, state: object) -> BatteryDelta:
        del state
        remaining_capacity = percept.config.battery_capacity - percept.battery_charge
        cheap_threshold = 0.14 + self._rng.uniform(-self.noise_scale, self.noise_scale)
        expensive_threshold = 0.34 + self._rng.uniform(-self.noise_scale, self.noise_scale)

        if percept.price <= cheap_threshold and remaining_capacity > 0.0:
            return BatteryDelta(min(percept.config.max_charge, remaining_capacity))

        if percept.price >= expensive_threshold and percept.battery_charge > self.reserve:
            return BatteryDelta(
                -min(percept.config.max_discharge, percept.battery_charge - self.reserve)
            )

        if percept.hour >= 21 and percept.battery_charge > 0.0:
            return BatteryDelta(-min(percept.config.max_discharge, percept.battery_charge))

        return BatteryDelta(0.0)
