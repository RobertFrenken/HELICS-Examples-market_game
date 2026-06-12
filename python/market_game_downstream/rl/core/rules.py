"""Compatibility exports for shared market-game rules.

Do not add new rules here. Add shared behavior to ``python.market_game_downstream.core``.
"""

from python.market_game_downstream.core.rules import (
    ClampResult,
    battery_delta_to_market_load,
    check_valid,
    clamp_market_load,
    compute_price_from_average_load,
    compute_price_from_total_load,
    ensure_valid,
    normalized_delta_to_market_load,
)
from python.market_game_downstream.rl.action_spaces import (
    BatteryPosture as BatteryAction,
    DEFAULT_ACTION_SPACE,
)


def action_to_market_load(action, base_demand, battery_charge, config=None):
    """Compatibility wrapper for the old coarse RL action mapping."""
    if config is None:
        from python.market_game_downstream.core.config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG
    return DEFAULT_ACTION_SPACE.market_load(
        action,
        base_demand,
        battery_charge,
        config,
    )

__all__ = [
    "BatteryAction",
    "ClampResult",
    "action_to_market_load",
    "battery_delta_to_market_load",
    "check_valid",
    "clamp_market_load",
    "compute_price_from_average_load",
    "compute_price_from_total_load",
    "ensure_valid",
    "normalized_delta_to_market_load",
]
