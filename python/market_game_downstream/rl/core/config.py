"""Compatibility exports for shared market-game configuration.

Do not add new rules here. Add shared behavior to ``python.market_game_downstream.core``.
"""

from python.market_game_downstream.core.config import (
    DEFAULT_CONFIG,
    PROFILE1_DEMAND,
    PROFILE_TYPES,
    MarketGameConfig,
    demand_profile,
)

__all__ = [
    "DEFAULT_CONFIG",
    "PROFILE1_DEMAND",
    "PROFILE_TYPES",
    "MarketGameConfig",
    "demand_profile",
]
