"""Configuration objects and default constants for the pure market simulator."""

from __future__ import annotations

from dataclasses import dataclass, field


PROFILE1_DEMAND = [
    2,
    1,
    1,
    1,
    2,
    4,
    6,
    8,
    9,
    7,
    5,
    4,
    3,
    4,
    5,
    7,
    9,
    12,
    10,
    7,
    5,
    4,
    2,
    2,
]


@dataclass(frozen=True)
class MarketGameConfig:
    """Rules/configuration for one market-game episode."""

    episode_hours: int = 24
    initial_price: float = 0.5
    initial_battery: float = 0.0
    battery_capacity: float = 20.0
    max_charge: float = 5.0
    max_discharge: float = 10.0
    demand_profile: list[float] = field(default_factory=lambda: list(PROFILE1_DEMAND))
    allow_negative_load: bool = True


DEFAULT_CONFIG = MarketGameConfig()
