"""Learner action spaces for RL/Gym adapters."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import BatteryDelta, BatteryPosture, BatteryPostureAction
from .percepts import MarketPercept


@dataclass(frozen=True)
class DiscreteBatteryPostureActionSpace:
    """Three-action smoke/debug baseline for battery posture."""

    def decode(self, action: object, percept: MarketPercept) -> BatteryPostureAction:
        del percept
        return BatteryPostureAction(BatteryPosture(action))


@dataclass(frozen=True)
class IntegerBatteryDeltaActionSpace:
    """Discrete battery-delta baseline with integer kWh actions."""

    min_delta: int = -10
    max_delta: int = 5

    def decode(self, action: object, percept: MarketPercept) -> BatteryDelta:
        del percept
        delta = int(action)
        if delta < self.min_delta or delta > self.max_delta:
            raise ValueError(
                f"invalid integer battery delta {action!r}; "
                f"expected {self.min_delta}..{self.max_delta}"
            )
        return BatteryDelta(float(delta))


@dataclass(frozen=True)
class ContinuousNormalizedDeltaActionSpace:
    """Continuous baseline: action in [-1, 1] maps to the legal delta range."""

    def decode(self, action: object, percept: MarketPercept) -> BatteryDelta:
        normalized = max(-1.0, min(1.0, float(action)))
        if normalized >= 0.0:
            return BatteryDelta(normalized * percept.config.max_charge)
        return BatteryDelta(normalized * percept.config.max_discharge)


DEFAULT_ACTION_SPACE = DiscreteBatteryPostureActionSpace()
