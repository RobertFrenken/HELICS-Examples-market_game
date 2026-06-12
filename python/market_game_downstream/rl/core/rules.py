"""Compatibility exports for shared market-game rules.

Do not add new rules here. Add shared behavior to ``python.market_game_downstream.core``.
"""

from python.market_game_downstream.core.rules import (
    BatteryAction,
    ClampResult,
    action_to_market_load,
    battery_delta_to_market_load,
    check_valid,
    clamp_market_load,
    compute_price_from_average_load,
    compute_price_from_total_load,
    ensure_valid,
    normalized_delta_to_market_load,
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
