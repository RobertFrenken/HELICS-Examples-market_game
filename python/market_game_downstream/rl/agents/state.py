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
