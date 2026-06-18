"""Optional learning records for market-game controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .percepts import MarketPercept


Observation = list[float]
InternalAction = Any


@dataclass(frozen=True)
class Experience:
    """Domain-level training/update record for adaptive controllers."""

    percept: MarketPercept
    action: object
    reward: float
    next_percept: MarketPercept
    terminated: bool
    info: dict[str, object] | None = None


@dataclass(frozen=True)
class Transition:
    """Vector-observation transition for RL libraries and learned controllers."""

    observation: Observation
    action: InternalAction
    reward: float
    next_observation: Observation
    terminated: bool
    info: dict[str, object] | None = None
