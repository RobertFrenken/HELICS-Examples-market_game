"""Observer classes for composed market-game agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from .contexts import MarketContext, Observation
from .interfaces import AgentState
from .observations import (
    InferenceBelief,
    InferenceFeatures,
    ObservationContext,
    ObservationMode,
    build_inference_observation,
    build_local_observation,
    build_observation,
    build_price_history_observation,
    update_inference_belief,
)


def observation_context(context: MarketContext) -> ObservationContext:
    return ObservationContext(
        hour=context.hour,
        price=context.price,
        battery_charge=context.battery_charge,
        demand=context.demand,
        price_history=context.price_history,
        own_market_load_history=context.own_market_load_history or [],
        house_count=context.house_count,
        config=context.config,
    )


@dataclass(frozen=True)
class LocalObserver:
    """Build local-only legal features."""

    def observe(self, context: MarketContext, state: AgentState) -> Observation:
        del state
        return build_local_observation(observation_context(context))


@dataclass(frozen=True)
class PriceHistoryObserver:
    """Build local features plus recent legal price-history summaries."""

    window: int = 6

    def observe(self, context: MarketContext, state: AgentState) -> Observation:
        del state
        return build_price_history_observation(
            observation_context(context),
            window=self.window,
        )


@dataclass
class InferenceObserver:
    """Build delayed aggregate-inference features with local observer state."""

    window: int = 6
    belief: InferenceBelief = field(default_factory=InferenceBelief)
    last_features: InferenceFeatures = field(default_factory=InferenceFeatures)

    def reset(self) -> None:
        self.belief.reset()
        self.last_features = InferenceFeatures()

    def observe(self, context: MarketContext, state: AgentState) -> Observation:
        del state
        obs_context = observation_context(context)
        self.last_features = update_inference_belief(obs_context, self.belief)
        return build_inference_observation(
            obs_context,
            self.last_features,
            window=self.window,
        )


@dataclass(frozen=True)
class ModeObserver:
    """Compatibility observer that delegates to the existing observation modes."""

    mode: ObservationMode | str = ObservationMode.PRICE_HISTORY

    def observe(self, context: MarketContext, state: AgentState) -> Observation:
        del state
        return build_observation(self.mode, observation_context(context))
