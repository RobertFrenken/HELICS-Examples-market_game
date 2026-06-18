"""Shared value objects for composed market-game agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from python.market_game_downstream.core import DEFAULT_CONFIG, MarketGameConfig


Observation = list[float]
InternalAction = Any


@dataclass(frozen=True)
class MarketPercept:
    """Legal information available to one house at one decision point."""

    price: float
    hour: int
    battery_charge: float
    demand: list[float]
    price_history: list[float]
    own_market_load_history: list[float] | None = None
    house_count: int = 1
    config: MarketGameConfig = DEFAULT_CONFIG

    @property
    def base_demand(self) -> float:
        return self.demand[self.hour]


MarketContext = MarketPercept


@dataclass(frozen=True)
class Experience:
    """Optional training/update record for adaptive strategies."""

    percept: MarketPercept
    action: object
    reward: float
    next_percept: MarketPercept
    terminated: bool
    info: dict[str, object] | None = None


@dataclass(frozen=True)
class Transition:
    """Vector-observation transition for RL libraries and learned controllers."""

    observation: Observation
    action: InternalAction
    reward: float
    next_observation: Observation
    terminated: bool
    info: dict[str, object] | None = None
