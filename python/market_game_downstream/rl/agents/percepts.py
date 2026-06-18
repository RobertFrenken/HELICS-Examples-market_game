"""Percepts available to market-game agents."""

from __future__ import annotations

from dataclasses import dataclass

from python.market_game_downstream.core import DEFAULT_CONFIG, MarketGameConfig


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
