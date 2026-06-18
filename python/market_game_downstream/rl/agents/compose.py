"""Composable policy object for model-agnostic market-game agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from .interfaces import (
    ActionProjector,
    AgentState,
    Controller,
    reset_if_supported,
)
from .percepts import MarketPercept
from .projectors import MarketActionProjector
from .state import NoAgentState


@dataclass
class MarketAgent:
    """Compose a controller and action projector behind compute_demand(...)."""

    name: str
    controller: Controller
    action_projector: ActionProjector = field(default_factory=MarketActionProjector)
    state: AgentState = field(default_factory=NoAgentState)

    def reset(self) -> None:
        self.state.reset()
        reset_if_supported(self.controller)
        reset_if_supported(self.action_projector)

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        percept = MarketPercept(
            price=price,
            hour=hour,
            battery_charge=battery_charge,
            demand=demand,
            price_history=price_history,
        )
        action = self.controller.decide(percept, self.state)
        return float(self.action_projector.market_load(action, percept))
