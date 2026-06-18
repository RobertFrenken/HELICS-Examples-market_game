"""Baseline policies for the pure market-game simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

from .compose import MarketAgent
from .controllers import (
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
from .state import InferenceBeliefState


@dataclass
class FollowDemandPolicy:
    """Passive baseline that submits the base demand profile exactly."""

    name: str = "FollowDemandHouse"
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = MarketAgent(
            name=self.name,
            controller=FollowDemandController(),
        )

    def reset(self) -> None:
        self._agent.reset()

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )


@dataclass
class InvalidDemandPolicy:
    """Stress opponent that deliberately submits illegal market loads."""

    name: str = "InvalidDemandHouse"
    load_offset: float = 100.0
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = MarketAgent(
            name=self.name,
            controller=InvalidDemandController(load_offset=self.load_offset),
        )

    def reset(self) -> None:
        self._agent.reset()

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )


@dataclass
class FlattenDemandPolicy:
    """Price-blind baseline that uses the battery to flatten own demand."""

    name: str = "FlattenDemandHouse"
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = MarketAgent(
            name=self.name,
            controller=FlattenDemandController(),
        )

    def reset(self) -> None:
        self._agent.reset()

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )


@dataclass
class FullCyclePolicy:
    """Mechanical baseline that cycles the battery between full and empty."""

    name: str = "FullCycleHouse"
    charging: bool = True
    _controller: FullCycleController = field(init=False, repr=False)
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._controller = FullCycleController(charging=self.charging)
        self._agent = MarketAgent(
            name=self.name,
            controller=self._controller,
        )

    def reset(self) -> None:
        self._agent.reset()
        self.charging = self._controller.charging

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        self._controller.charging = self.charging
        load = self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )
        self.charging = self._controller.charging
        return load


@dataclass
class PriceAwarePolicy:
    """Threshold policy with simple price bands and time-of-day reserves."""

    name: str = "PriceAwareHouse"
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = MarketAgent(
            name=self.name,
            controller=PriceAwareController(),
        )

    def reset(self) -> None:
        self._agent.reset()

    def reserve_target(self, hour: int) -> float:
        return PriceAwareController().reserve_target(hour)

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )


@dataclass
class RollingPricePolicy:
    """Legal adaptive policy that compares price to a rolling recent mean."""

    name: str = "RollingPriceHouse"
    window: int = 6
    cheap_ratio: float = 0.94
    expensive_ratio: float = 1.08
    reserve: float = 4.0
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = MarketAgent(
            name=self.name,
            controller=RollingThresholdController(
                window=self.window,
                cheap_ratio=self.cheap_ratio,
                expensive_ratio=self.expensive_ratio,
                reserve=self.reserve,
            ),
        )

    def reset(self) -> None:
        self._agent.reset()

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )


@dataclass
class LegalInferencePolicy:
    """Legal-observation policy that infers aggregate pressure from prices."""

    name: str = "LegalInferenceHouse"
    house_count: int = 3
    belief: InferenceBeliefState = field(default_factory=InferenceBeliefState)
    _controller: LegalInferenceController = field(init=False, repr=False)
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._controller = LegalInferenceController(
            house_count=self.house_count,
        )
        self._agent = MarketAgent(
            name=self.name,
            controller=self._controller,
            state=self.belief,
        )

    def reset(self) -> None:
        self._agent.reset()

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        load = self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )
        return load


@dataclass
class NoisyThresholdPolicy:
    """Seeded threshold opponent with repeatable per-hour jitter."""

    name: str = "NoisyThresholdHouse"
    seed: int = 1
    reserve: float = 4.0
    noise_scale: float = 0.04
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = MarketAgent(
            name=self.name,
            controller=NoisyThresholdController(
                seed=self.seed,
                reserve=self.reserve,
                noise_scale=self.noise_scale,
            ),
        )

    def reset(self) -> None:
        self._agent.reset()

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )


@dataclass
class OscillatingPolicy:
    """Price-blind opponent with sinusoidal charge/discharge swings."""

    name: str = "OscillatingHouse"
    period: int = 4
    phase: int = 0
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = MarketAgent(
            name=self.name,
            controller=OscillatingController(period=self.period, phase=self.phase),
        )

    def reset(self) -> None:
        self._agent.reset()

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )


@dataclass
class VolatilitySeekingPolicy:
    """Chaotic opponent that tends to amplify price movement."""

    name: str = "VolatilitySeekingHouse"
    _agent: MarketAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = MarketAgent(
            name=self.name,
            controller=VolatilitySeekingController(),
        )

    def reset(self) -> None:
        self._agent.reset()

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return self._agent.compute_demand(
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )


POLICY_TYPES = {
    "FlattenDemandPolicy": FlattenDemandPolicy,
    "FollowDemandPolicy": FollowDemandPolicy,
    "FullCyclePolicy": FullCyclePolicy,
    "InvalidDemandPolicy": InvalidDemandPolicy,
    "LegalInferencePolicy": LegalInferencePolicy,
    "NoisyThresholdPolicy": NoisyThresholdPolicy,
    "OscillatingPolicy": OscillatingPolicy,
    "PriceAwarePolicy": PriceAwarePolicy,
    "RollingPricePolicy": RollingPricePolicy,
    "VolatilitySeekingPolicy": VolatilitySeekingPolicy,
}
