"""State containers for composed market-game agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NoAgentState:
    """Default empty state for stateless composed agents."""

    def reset(self) -> None:
        pass


@dataclass
class DictAgentState:
    """Small mutable state bag for simple adaptive strategies."""

    values: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.values.clear()


@dataclass
class InferenceBeliefState:
    """Belief state for delayed aggregate-load inference."""

    own_load_history: list[float] = field(default_factory=list)
    crowd_battery: float = 0.0

    def reset(self) -> None:
        self.own_load_history.clear()
        self.crowd_battery = 0.0
