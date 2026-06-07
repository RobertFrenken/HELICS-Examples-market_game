"""Competition scenario helpers for downstream market-game experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import random

from python.market_game_downstream.core import (
    DEFAULT_CONFIG,
    MarketGameConfig,
    MarketScenario,
    demand_profile,
)
from python.market_game_downstream.core.simulator import HousePolicy, run_scenario
from .agents.policies import (
    FlattenDemandPolicy,
    FollowDemandPolicy,
    FullCyclePolicy,
    LegalInferencePolicy,
    NoisyThresholdPolicy,
    OscillatingPolicy,
    PriceAwarePolicy,
    RollingPricePolicy,
    VolatilitySeekingPolicy,
)


PolicyFactory = Callable[[], HousePolicy]


@dataclass(frozen=True)
class CompetitionScenario:
    """Configurable weekly scenario for local training and evaluation."""

    name: str
    profile_type: str = "profile1"
    seed: int = 1
    policy_factories: list[PolicyFactory] = field(default_factory=list)
    config: MarketGameConfig = DEFAULT_CONFIG

    def demand(self) -> list[float]:
        return demand_profile(self.profile_type, rng=random.Random(self.seed))

    def policies(self) -> list[HousePolicy]:
        return [factory() for factory in self.policy_factories]

    def to_market_scenario(self) -> MarketScenario:
        return MarketScenario(
            policies=self.policies(),
            demand_profile=self.demand(),
            config=self.config,
        )


def stock_example_scenario() -> CompetitionScenario:
    return CompetitionScenario(
        name="stock_examples_profile1",
        profile_type="profile1",
        policy_factories=[
            FlattenDemandPolicy,
            FullCyclePolicy,
            PriceAwarePolicy,
        ],
    )


def weekly_training_scenarios(seed: int = 1) -> list[CompetitionScenario]:
    """Return a small curriculum matching the competition progression."""
    return [
        CompetitionScenario(
            name="week_1_baselines",
            profile_type="profile1",
            seed=seed,
            policy_factories=[
                FlattenDemandPolicy,
                PriceAwarePolicy,
                RollingPricePolicy,
            ],
        ),
        CompetitionScenario(
            name="week_2_new_profile",
            profile_type="random",
            seed=seed,
            policy_factories=[
                PriceAwarePolicy,
                RollingPricePolicy,
                lambda: NoisyThresholdPolicy(seed=seed),
            ],
        ),
        CompetitionScenario(
            name="week_3_mixed_population",
            profile_type="spike",
            seed=seed,
            policy_factories=[
                FollowDemandPolicy,
                LegalInferencePolicy,
                lambda: NoisyThresholdPolicy(seed=seed + 1),
                lambda: OscillatingPolicy(phase=1),
            ],
        ),
        CompetitionScenario(
            name="week_4_chaotic_houses",
            profile_type="dspike",
            seed=seed,
            policy_factories=[
                PriceAwarePolicy,
                lambda: NoisyThresholdPolicy(seed=seed + 2, noise_scale=0.08),
                lambda: OscillatingPolicy(period=3),
                VolatilitySeekingPolicy,
            ],
        ),
    ]


def evaluate_scenario(scenario: CompetitionScenario) -> list[dict[str, str]]:
    """Run one scenario and return CSV-friendly summary rows."""
    result = run_scenario(scenario.to_market_scenario())
    return [_summary_row(scenario, house) for house in result.houses]


def evaluate_curriculum(seed: int = 1) -> list[dict[str, str]]:
    return [
        row
        for scenario in weekly_training_scenarios(seed=seed)
        for row in evaluate_scenario(scenario)
    ]


def _summary_row(scenario: CompetitionScenario, house: object) -> dict[str, str]:
    return {
        "scenario": scenario.name,
        "agent": house.policy.name,
        "profile_type": scenario.profile_type,
        "seed": str(scenario.seed),
        "total_load": f"{house.total_load:.10f}",
        "total_cost": f"{house.total_cost:.10f}",
        "final_battery": f"{house.battery.energy:.10f}",
        "boundary_warnings": str(len(house.boundary_warnings)),
        "clamps": str(house.clamps),
    }
