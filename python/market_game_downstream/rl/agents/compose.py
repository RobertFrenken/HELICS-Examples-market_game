"""Composable policy object for model-agnostic market-game agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from .contexts import MarketPercept
from .interfaces import (
    ActionProjector,
    Actuator,
    AgentState,
    Controller,
    Observer,
    Strategy,
    reset_if_supported,
)
from .projectors import MarketActionProjector
from .state import NoAgentState


@dataclass
class MarketAgent:
    """Compose observation, strategy, and actuation behind compute_demand(...)."""

    name: str
    controller: Controller | None = None
    action_projector: ActionProjector = field(default_factory=MarketActionProjector)
    observer: Observer | None = None
    strategy: Strategy | None = None
    actuator: Actuator | None = None
    state: AgentState = field(default_factory=NoAgentState)

    def __post_init__(self) -> None:
        if self.controller is None and (
            self.observer is None or self.strategy is None or self.actuator is None
        ):
            raise ValueError(
                "MarketAgent requires either controller=... or "
                "observer=..., strategy=..., and actuator=..."
            )

    def reset(self) -> None:
        self.state.reset()
        reset_if_supported(self.controller)
        reset_if_supported(self.action_projector)
        if self.observer is not None:
            reset_if_supported(self.observer)
        if self.strategy is not None:
            reset_if_supported(self.strategy)
        if self.actuator is not None:
            reset_if_supported(self.actuator)

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
        if self.controller is not None:
            action = self.controller.decide(percept, self.state)
            return float(self.action_projector.market_load(action, percept))

        assert self.observer is not None
        assert self.strategy is not None
        assert self.actuator is not None
        observation = self.observer.observe(percept, self.state)
        action = self.strategy.decide(observation, self.state)
        return float(self.actuator.market_load(action, percept))
