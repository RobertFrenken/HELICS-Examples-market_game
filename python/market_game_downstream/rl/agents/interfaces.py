"""Protocols for model-agnostic market-game agent composition."""

from __future__ import annotations

from typing import Protocol

from python.market_game_downstream.core import DEFAULT_CONFIG, MarketGameConfig
from .contexts import (
    Experience,
    InternalAction,
    MarketContext,
    MarketPercept,
    Observation,
    Transition,
)
from .market_actions import MarketAction


class AgentState(Protocol):
    """Episode-local state carried by a composed agent."""

    def reset(self) -> None:
        """Reset episode-local state."""


class Observer(Protocol):
    """Convert legal market context into strategy-facing observations."""

    def observe(self, context: MarketContext, state: AgentState) -> Observation:
        """Return the observation for one decision point."""


class Strategy(Protocol):
    """Choose an internal action from an observation."""

    def decide(self, observation: Observation, state: AgentState) -> InternalAction:
        """Return a strategy-specific action."""


class Actuator(Protocol):
    """Project an internal action onto a legal market load."""

    def market_load(self, action: InternalAction, context: MarketContext) -> float:
        """Return the proposed market load for one decision point."""


class Controller(Protocol):
    """Map legal market percepts and agent state to semantic market actions."""

    def decide(self, percept: MarketPercept, state: AgentState) -> MarketAction:
        """Return the next semantic market action."""


class ActionProjector(Protocol):
    """Project semantic market actions onto the simulator's market-load ABI."""

    def market_load(self, action: MarketAction, percept: MarketPercept) -> float:
        """Return the proposed market load for one decision point."""


class FeatureExtractor(Protocol):
    """Encode a market percept for vector-based controllers."""

    def encode(self, percept: MarketPercept, state: AgentState) -> Observation:
        """Return a numeric observation for a vector-based controller."""


class ActionMapper(Protocol):
    """Compatibility protocol for legacy RL action mappers."""

    def market_load(
        self,
        action: object,
        base_demand: float,
        battery_charge: float,
        config: MarketGameConfig = DEFAULT_CONFIG,
    ) -> float:
        """Return the proposed market load for one learner action."""


class Tunable(Protocol):
    """Optional capability for algorithms with externally tunable parameters."""

    def get_params(self) -> dict[str, object]:
        """Return tunable constructor-style parameters."""

    def set_params(self, **params: object) -> None:
        """Update tunable parameters."""


class Trainable(Protocol):
    """Optional capability for strategies that learn from transitions."""

    def update(self, transition: Transition) -> None:
        """Update strategy state from one transition."""


class ExperienceLearner(Protocol):
    """Optional capability for controllers that learn from domain experiences."""

    def update(self, experience: Experience) -> None:
        """Update controller state from one domain experience."""


def reset_if_supported(component: object) -> None:
    """Reset a component if it exposes a no-argument reset method."""
    reset = getattr(component, "reset", None)
    if reset is not None:
        reset()
