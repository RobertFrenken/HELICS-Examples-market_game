"""Compatibility battery helpers for the HELICS market game.

Do not add new market rules here. Add shared behavior to
``python.market_game_downstream.core`` and import it here for older house files.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.market_game_downstream.core import DEFAULT_CONFIG
from python.market_game_downstream.core.rules import check_valid, ensure_valid
from python.market_game_downstream.core.simulator import BatteryState


BATTERY_CAPCITY = DEFAULT_CONFIG.battery_capacity
BATTERY_CAPACITY = DEFAULT_CONFIG.battery_capacity
BATTERY_MAX_DISCHARGE = DEFAULT_CONFIG.max_discharge
BATTERY_MAX_CHARGE = DEFAULT_CONFIG.max_charge


class Battery(BatteryState):
    """Original HELICS battery API backed by shared core rules."""

    def __init__(self, energy: float = 0):
        super().__init__(energy=energy, config=DEFAULT_CONFIG)

    def discharge(self, delta: float) -> float:
        return self.change(-abs(delta))

    def charge(self, delta: float) -> float:
        return self.change(abs(delta))
