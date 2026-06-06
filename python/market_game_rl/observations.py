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


LOCAL_OBSERVATION_NAMES = [
    "hour_sin",
    "hour_cos",
    "current_price",
    "own_battery_charge",
    "own_base_demand_now",
]

PRICE_HISTORY_OBSERVATION_NAMES = LOCAL_OBSERVATION_NAMES + [
    "price_lag_1",
    "price_lag_2",
    "price_mean_recent",
    "price_trend_recent",
    "price_volatility_recent",
]

INFERENCE_OBSERVATION_NAMES = PRICE_HISTORY_OBSERVATION_NAMES + [
    "inferred_avg_market_load_lag_1",
    "inferred_others_avg_load_lag_1",
    "inferred_crowd_delta_lag_1",
    "estimated_crowd_battery",
    "tier_distance",
    "inverse_uncertainty",
]

OBSERVATION_SCHEMAS = {
    ObservationMode.LOCAL: LOCAL_OBSERVATION_NAMES,
    ObservationMode.PRICE_HISTORY: PRICE_HISTORY_OBSERVATION_NAMES,
    ObservationMode.INFERENCE: INFERENCE_OBSERVATION_NAMES,
}


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


@dataclass(frozen=True)
class InferenceFeatures:
    inferred_average: float = 0.0
    inferred_others_average: float = 0.0
    inferred_crowd_delta: float = 0.0
    crowd_battery: float = 0.0
    tier_distance: float = 999.0
    inverse_uncertainty: float = 0.0


def observation_schema(mode: ObservationMode | str) -> list[str]:
    return list(OBSERVATION_SCHEMAS[ObservationMode(mode)])


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
    features: InferenceFeatures,
    window: int = 6,
) -> list[float]:
    """Build price-history features plus delayed aggregate inference features."""
    return build_price_history_observation(context, window=window) + [
        features.inferred_average,
        features.inferred_others_average,
        features.inferred_crowd_delta,
        features.crowd_battery,
        features.tier_distance,
        features.inverse_uncertainty,
    ]


def update_inference_belief(
    context: ObservationContext,
    belief: InferenceBelief,
) -> InferenceFeatures:
    """Update delayed aggregate belief and return inference features.

    This is intentionally separate from observation vector construction so the
    state mutation is explicit at the environment boundary.
    """
    if context.hour <= 0 or not context.own_market_load_history:
        return InferenceFeatures(crowd_battery=belief.crowd_battery)

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

    return InferenceFeatures(
        inferred_average=inferred_average,
        inferred_others_average=inferred_others_average,
        inferred_crowd_delta=inferred_crowd_delta,
        crowd_battery=belief.crowd_battery,
        tier_distance=distance_to_nearest_pricing_threshold(inferred_average),
        inverse_uncertainty=inverse.uncertainty,
    )


def build_observation(
    mode: ObservationMode | str,
    context: ObservationContext,
    inference_features: InferenceFeatures | None = None,
) -> list[float]:
    mode = ObservationMode(mode)
    if mode == ObservationMode.LOCAL:
        return build_local_observation(context)
    if mode == ObservationMode.PRICE_HISTORY:
        return build_price_history_observation(context)
    if inference_features is None:
        inference_features = InferenceFeatures()
    return build_inference_observation(context, inference_features)
