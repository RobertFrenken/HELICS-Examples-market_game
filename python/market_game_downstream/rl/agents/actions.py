"""Semantic market actions emitted by market-game controllers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class BatteryPosture(IntEnum):
    """Coarse charge/neutral/discharge battery posture."""

    DISCHARGE = -1
    NEUTRAL = 0
    CHARGE = 1


@dataclass(frozen=True)
class FollowDemand:
    """Submit the current base demand."""


@dataclass(frozen=True)
class TargetLoad:
    """Submit an explicit market load."""

    load: float


@dataclass(frozen=True)
class BatteryDelta:
    """Charge or discharge the battery by a signed kWh delta."""

    kwh: float


@dataclass(frozen=True)
class BatteryPostureAction:
    """Use a coarse charge/neutral/discharge battery posture."""

    posture: BatteryPosture


MarketAction = FollowDemand | TargetLoad | BatteryDelta | BatteryPostureAction | float
