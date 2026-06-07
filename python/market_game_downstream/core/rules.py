"""Market-game rule functions shared by HELICS scripts, simulation, and RL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Protocol

from .config import DEFAULT_CONFIG, MarketGameConfig


class BatteryLike(Protocol):
    def current_charge(self) -> float:
        """Return current stored energy."""


class BatteryAction(IntEnum):
    """Discrete battery posture for student examples and RL control."""

    DISCHARGE = -1
    NEUTRAL = 0
    CHARGE = 1


@dataclass(frozen=True)
class ClampResult:
    market_load: float
    warning: str = ""


def compute_price_from_average_load(average_market_load: float) -> float:
    """Return the market price for one average market-facing load value."""
    _finite_number(average_market_load, "average_market_load")
    m = average_market_load
    if m < 3.0:
        return 0.1
    if m < 6.0:
        return 0.1 + 0.03 * (m - 3.0)
    if m < 9.0:
        return 0.19 + 0.1 * (m - 6.0)
    if m < 13.0:
        return 0.49 + 0.25 * (m - 9.0)
    return 1.49 + 1.0 * (m - 13.0)


def compute_price_from_total_load(total_market_load: float, house_count: int) -> float:
    _finite_number(total_market_load, "total_market_load")
    if house_count < 0:
        raise ValueError("house_count must be >= 0")
    if house_count == 0:
        return 0.1
    return compute_price_from_average_load(total_market_load / house_count)


def clamp_market_load(
    market_load: float,
    base_demand: float,
    battery_charge: float,
    config: MarketGameConfig = DEFAULT_CONFIG,
) -> ClampResult:
    """Clamp a submitted market load to the game's battery constraints."""
    _finite_number(market_load, "market_load")
    _finite_number(base_demand, "base_demand")
    _finite_number(battery_charge, "battery_charge")
    if market_load >= base_demand + (config.battery_capacity - battery_charge):
        return ClampResult(
            base_demand + (config.battery_capacity - battery_charge),
            "listed battery charge rate exceeds available battery storage capacity",
        )
    if market_load >= base_demand + config.max_charge:
        return ClampResult(
            base_demand + config.max_charge,
            "listed battery charge rate exceeds maximum charge rate",
        )
    if market_load <= base_demand - battery_charge:
        return ClampResult(
            base_demand - battery_charge,
            "listed consumption exceeds available battery energy",
        )
    if market_load <= base_demand - config.max_discharge:
        return ClampResult(
            base_demand - config.max_discharge,
            "listed consumption exceeds max battery discharge rate",
        )
    if not config.allow_negative_load and market_load < 0.0:
        return ClampResult(0.0, "listed market load is negative")
    return ClampResult(market_load)


def action_to_market_load(
    action: BatteryAction | int,
    base_demand: float,
    battery_charge: float,
    config: MarketGameConfig = DEFAULT_CONFIG,
) -> float:
    """Convert a discrete battery action into a legal market-facing load."""
    action = BatteryAction(action)
    if action == BatteryAction.DISCHARGE:
        proposed = base_demand - config.max_discharge
    elif action == BatteryAction.NEUTRAL:
        proposed = base_demand
    else:
        proposed = base_demand + config.max_charge
    return clamp_market_load(proposed, base_demand, battery_charge, config).market_load


def _battery_charge(battery: BatteryLike | float) -> float:
    if isinstance(battery, int | float):
        return float(battery)
    return battery.current_charge()


def _finite_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")


def ensure_valid(
    value: float,
    base_demand: float,
    battery: BatteryLike | float,
    config: MarketGameConfig = DEFAULT_CONFIG,
) -> float:
    """Compatibility wrapper for the original ``battery.ensure_valid`` helper."""
    return clamp_market_load(value, base_demand, _battery_charge(battery), config).market_load


def check_valid(
    value: float,
    base_demand: float,
    battery: BatteryLike | float,
    config: MarketGameConfig = DEFAULT_CONFIG,
) -> str:
    """Compatibility wrapper for the original ``battery.check_valid`` helper."""
    return clamp_market_load(value, base_demand, _battery_charge(battery), config).warning
