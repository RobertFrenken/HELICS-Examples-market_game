"""Configuration objects and demand profiles for the market game."""

from __future__ import annotations

from dataclasses import dataclass, field
import random


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

SOLAR_DEMAND_BASE = [
    2,
    2,
    2,
    2,
    3,
    4,
    5,
    2,
    -4,
    -6,
    -7,
    -8,
    -7,
    -6,
    -3,
    1,
    4,
    8,
    10,
    11,
    10,
    7,
    5,
    3,
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


def demand_profile(profile_type: str, rng: random.Random | None = None) -> list[float]:
    """Return a 24-hour base demand profile matching the HELICS market maker."""
    rng = rng or random
    if profile_type == "random":
        elements = [rng.random() for _ in range(24)]
        multiplier = 120.0 / sum(elements)
        return [value * multiplier for value in elements]
    if profile_type == "spike":
        profile = [4] * 24
        profile[rng.randint(0, 23)] = 28
        return profile
    if profile_type == "dspike":
        profile = [3] * 24
        profile[rng.randint(0, 23)] += 24
        profile[rng.randint(0, 23)] += 24
        return profile
    if profile_type == "profile1":
        return list(PROFILE1_DEMAND)
    if profile_type == "profile_solar":
        return [value * 3.0 for value in SOLAR_DEMAND_BASE]
    return [5] * 24
