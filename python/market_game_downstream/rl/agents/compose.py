"""Composable policy object for model-agnostic market-game agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from .primitives import reset_if_supported
from .primitives import MarketPercept
from .projectors import project_market_load
from .state import NoAgentState


@dataclass
class MarketAgent:
    """Compose a controller and action projector behind compute_demand(...)."""

    name: str
    controller: object
    state: object = field(default_factory=NoAgentState)

    def reset(self) -> None:
        self.state.reset()
        reset_if_supported(self.controller)

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
        return float(project_market_load(action, percept))
