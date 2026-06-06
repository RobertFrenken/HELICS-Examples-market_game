"""Legal observation builders for RL-style market-game policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .config import DEFAULT_CONFIG, MarketGameConfig
from .features import (
    distance_to_nearest_pricing_threshold,
    estimate_others_average_load,
    invert_price_to_average_load,
    recent_mean,
    recent_volatility,
)


class ObservationMode(str, Enum):
    LOCAL = "local"
    PRICE_HISTORY = "price_history"
    INFERENCE = "inference"


@dataclass
class ObservationContext:
    """Legal information available to a house at one decision point."""

    hour: int
    price: float
    battery_charge: float
    demand: list[float]
    price_history: list[float]
    own_market_load_history: list[float]
    house_count: int
    config: MarketGameConfig = DEFAULT_CONFIG

    @property
    def base_demand(self) -> float:
        return self.demand[self.hour]


@dataclass
class InferenceBelief:
    """State carried between observations for aggregate inference features."""

    crowd_battery: float = 0.0

    def reset(self) -> None:
        self.crowd_battery = 0.0


def build_local_observation(context: ObservationContext) -> list[float]:
    """Build local-only legal features.

    Features:

    ``hour_sin = sin(2*pi*hour/episode_hours)``
    ``hour_cos = cos(2*pi*hour/episode_hours)``
    ``current_price = p_t``
    ``own_battery_charge = B_t``
    ``own_base_demand_now = D_t``
    """
    angle = 2.0 * math.pi * context.hour / context.config.episode_hours
    return [
        math.sin(angle),
        math.cos(angle),
        context.price,
        context.battery_charge,
        context.base_demand,
    ]


def build_price_history_observation(context: ObservationContext, window: int = 6) -> list[float]:
    """Build local features plus legal price-history summaries."""
    previous_prices = context.price_history[:-1]
    price_lag_1 = previous_prices[-1] if len(previous_prices) >= 1 else context.price
    price_lag_2 = previous_prices[-2] if len(previous_prices) >= 2 else context.price
    price_mean = recent_mean(previous_prices, fallback=context.price, window=window)
    price_trend = (
        previous_prices[-1] - previous_prices[-2]
        if len(previous_prices) >= 2
        else 0.0
    )
    price_volatility = recent_volatility(previous_prices, window=window)

    return build_local_observation(context) + [
        price_lag_1,
        price_lag_2,
        price_mean,
        price_trend,
        price_volatility,
    ]


def build_inference_observation(
    context: ObservationContext,
    belief: InferenceBelief,
    window: int = 6,
) -> list[float]:
    """Build price-history features plus delayed aggregate inference features."""
    inferred_average = 0.0
    inferred_others_average = 0.0
    inferred_crowd_delta = 0.0
    tier_distance = 999.0
    inverse_uncertainty = 0.0

    if context.hour > 0 and context.own_market_load_history:
        inverse = invert_price_to_average_load(context.price)
        inferred_average = inverse.estimate
        previous_own_load = context.own_market_load_history[-1]
        inferred_others_average = estimate_others_average_load(
            inferred_average,
            previous_own_load,
            context.house_count,
        )
        previous_base_demand = context.demand[context.hour - 1]
        inferred_crowd_delta = inferred_others_average - previous_base_demand
        belief.crowd_battery = min(
            context.config.battery_capacity,
            max(0.0, belief.crowd_battery + inferred_crowd_delta),
        )
        tier_distance = distance_to_nearest_pricing_threshold(inferred_average)
        inverse_uncertainty = inverse.uncertainty

    return build_price_history_observation(context, window=window) + [
        inferred_average,
        inferred_others_average,
        inferred_crowd_delta,
        belief.crowd_battery,
        tier_distance,
        inverse_uncertainty,
    ]


def build_observation(
    mode: ObservationMode | str,
    context: ObservationContext,
    belief: InferenceBelief | None = None,
) -> list[float]:
    mode = ObservationMode(mode)
    if mode == ObservationMode.LOCAL:
        return build_local_observation(context)
    if mode == ObservationMode.PRICE_HISTORY:
        return build_price_history_observation(context)
    if belief is None:
        belief = InferenceBelief()
    return build_inference_observation(context, belief)
