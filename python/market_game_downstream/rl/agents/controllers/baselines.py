"""Baseline controllers for market-game agents."""

from __future__ import annotations

from dataclasses import dataclass

from ..contexts import MarketPercept
from ..interfaces import AgentState
from ..market_actions import FollowDemand


@dataclass(frozen=True)
class FollowDemandController:
    """Passive controller that submits the current base demand."""

    def decide(self, percept: MarketPercept, state: AgentState) -> FollowDemand:
        del percept, state
        return FollowDemand()
