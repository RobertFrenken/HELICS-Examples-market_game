"""Protocols for model-agnostic market-game agent composition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

from .actions import MarketAction
from .experiences import (
    Experience,
    Observation,
    Transition,
)
from .percepts import MarketPercept


ControllerActionT = TypeVar("ControllerActionT", bound=MarketAction, covariant=True)


class AgentState(Protocol):
    """Episode-local state carried by a composed agent."""

    def reset(self) -> None:
        """Reset episode-local state."""


class Controller(Protocol[ControllerActionT]):
    """Map legal market percepts and agent state to semantic market actions."""

    def decide(self, percept: MarketPercept, state: AgentState) -> ControllerActionT:
        """Return the next semantic market action."""


class BaseController(Generic[ControllerActionT], ABC):
    """Generic base class for concrete market-game controllers."""

    @abstractmethod
    def decide(self, percept: MarketPercept, state: AgentState) -> ControllerActionT:
        """Return the next semantic market action."""


class ActionProjector(Protocol):
    """Project semantic market actions onto the simulator's market-load ABI."""

    def market_load(self, action: MarketAction, percept: MarketPercept) -> float:
        """Return the proposed market load for one decision point."""


class FeatureExtractor(Protocol):
    """Encode a market percept for vector-based controllers."""

    def encode(self, percept: MarketPercept, state: AgentState) -> Observation:
        """Return a numeric observation for a vector-based controller."""


class BaseFeatureExtractor(ABC):
    """Generic base class for concrete vector feature extractors."""

    @abstractmethod
    def encode(self, percept: MarketPercept, state: AgentState) -> Observation:
        """Return a numeric observation for a vector-based controller."""


class LearnerActionSpace(Protocol):
    """Decode learner actions into semantic market actions."""

    def decode(self, action: object, percept: MarketPercept) -> MarketAction:
        """Return the semantic market action represented by one learner action."""


class Tunable(Protocol):
    """Optional capability for algorithms with externally tunable parameters."""

    def get_params(self) -> dict[str, object]:
        """Return tunable constructor-style parameters."""

    def set_params(self, **params: object) -> None:
        """Update tunable parameters."""


class Trainable(Protocol):
    """Optional capability for controllers that learn from vector transitions."""

    def update(self, transition: Transition) -> None:
        """Update controller state from one transition."""


class ExperienceLearner(Protocol):
    """Optional capability for controllers that learn from domain experiences."""

    def update(self, experience: Experience) -> None:
        """Update controller state from one domain experience."""


def reset_if_supported(component: object) -> None:
    """Reset a component if it exposes a no-argument reset method."""
    reset = getattr(component, "reset", None)
    if reset is not None:
        reset()
