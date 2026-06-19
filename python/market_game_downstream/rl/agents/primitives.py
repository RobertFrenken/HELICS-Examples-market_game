"""Core market-agent primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

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


class BatteryPosture(IntEnum):
    """Coarse charge/neutral/discharge battery posture."""

    DISCHARGE = -1
    NEUTRAL = 0
    CHARGE = 1


@dataclass(frozen=True)
class TargetLoad:
    """Submit an explicit market load."""

    load: float


@dataclass(frozen=True)
class BatteryDelta:
    """Charge or discharge the battery by a signed kWh delta."""

    kwh: float


MarketAction: TypeAlias = TargetLoad | BatteryDelta | BatteryPosture | float


def reset_if_supported(component: object) -> None:
    """Reset a component if it exposes a no-argument reset method."""
    reset = getattr(component, "reset", None)
    if reset is not None:
        reset()
