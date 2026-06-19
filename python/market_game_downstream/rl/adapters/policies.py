"""Factory helpers for simulator-facing controller policies."""

from __future__ import annotations

from ..agents.compose import MarketAgent
from ..agents.controllers import (
    FlattenDemandController,
    FollowDemandController,
    FullCycleController,
    InvalidDemandController,
    LegalInferenceController,
    NoisyThresholdController,
    OscillatingController,
    PriceAwareController,
    RollingThresholdController,
    VolatilitySeekingController,
)
from ..agents.state import InferenceBeliefState, NoAgentState


def controller_policy(
    name: str,
    controller: object,
    state: object | None = None,
) -> MarketAgent:
    """Adapt a controller to the simulator's HousePolicy shape."""
    return MarketAgent(
        name=name,
        controller=controller,
        state=state if state is not None else NoAgentState(),
    )


def FollowDemandPolicy(name: str = "FollowDemandHouse") -> MarketAgent:
    return controller_policy(name, FollowDemandController())


def InvalidDemandPolicy(
    name: str = "InvalidDemandHouse",
    load_offset: float = 100.0,
) -> MarketAgent:
    return controller_policy(name, InvalidDemandController(load_offset=load_offset))


def FlattenDemandPolicy(name: str = "FlattenDemandHouse") -> MarketAgent:
    return controller_policy(name, FlattenDemandController())


def FullCyclePolicy(
    name: str = "FullCycleHouse",
    charging: bool = True,
) -> MarketAgent:
    return controller_policy(name, FullCycleController(charging=charging))


def PriceAwarePolicy(name: str = "PriceAwareHouse") -> MarketAgent:
    return controller_policy(name, PriceAwareController())


def RollingPricePolicy(
    name: str = "RollingPriceHouse",
    window: int = 6,
    cheap_ratio: float = 0.94,
    expensive_ratio: float = 1.08,
    reserve: float = 4.0,
) -> MarketAgent:
    return controller_policy(
        name,
        RollingThresholdController(
            window=window,
            cheap_ratio=cheap_ratio,
            expensive_ratio=expensive_ratio,
            reserve=reserve,
        ),
    )


def LegalInferencePolicy(
    name: str = "LegalInferenceHouse",
    house_count: int = 3,
    belief: InferenceBeliefState | None = None,
) -> MarketAgent:
    return controller_policy(
        name,
        LegalInferenceController(house_count=house_count),
        state=belief if belief is not None else InferenceBeliefState(),
    )


def NoisyThresholdPolicy(
    name: str = "NoisyThresholdHouse",
    seed: int = 1,
    reserve: float = 4.0,
    noise_scale: float = 0.04,
) -> MarketAgent:
    return controller_policy(
        name,
        NoisyThresholdController(
            seed=seed,
            reserve=reserve,
            noise_scale=noise_scale,
        ),
    )


def OscillatingPolicy(
    name: str = "OscillatingHouse",
    period: int = 4,
    phase: int = 0,
) -> MarketAgent:
    return controller_policy(name, OscillatingController(period=period, phase=phase))


def VolatilitySeekingPolicy(name: str = "VolatilitySeekingHouse") -> MarketAgent:
    return controller_policy(name, VolatilitySeekingController())
