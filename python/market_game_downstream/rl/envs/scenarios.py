"""Built-in market-game scenarios for RL environments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import random

from python.market_game_downstream.core import (
    DEFAULT_CONFIG,
    HousePolicy,
    MarketGameConfig,
    MarketScenario,
    demand_profile,
)
from python.market_game_downstream.rl.agents.policies import (
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
    name: str
    profile_type: str = "profile1"
    seed: int = 1
    policy_factories: list[PolicyFactory] = field(default_factory=list)
    config: MarketGameConfig = DEFAULT_CONFIG

    def demand(self) -> list[float]:
        return demand_profile(self.profile_type, rng=random.Random(self.seed))

    def policies(self) -> list[HousePolicy]:
        policies = [factory() for factory in self.policy_factories]
        _validate_unique_policy_names(self.name, policies)
        return policies

    def to_market_scenario(self) -> MarketScenario:
        return MarketScenario(
            policies=self.policies(),
            demand_profile=self.demand(),
            config=self.config,
        )

    def to_env_config(self) -> tuple[list[HousePolicy], MarketGameConfig]:
        return (
            self.policies(),
            replace(self.config, demand_profile=self.demand()),
        )

    def with_policy_factory(
        self,
        policy_factory: PolicyFactory,
        *,
        first: bool = True,
    ) -> "CompetitionScenario":
        factories = (
            [policy_factory, *self.policy_factories]
            if first
            else [*self.policy_factories, policy_factory]
        )
        return replace(self, policy_factories=factories)


def stock_example_scenario(seed: int = 1) -> CompetitionScenario:
    return CompetitionScenario(
        name="stock_examples_profile1",
        profile_type="profile1",
        seed=seed,
        policy_factories=[
            FlattenDemandPolicy,
            FullCyclePolicy,
            PriceAwarePolicy,
        ],
    )


def weekly_training_scenarios(seed: int = 1) -> list[CompetitionScenario]:
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
                lambda seed=seed: NoisyThresholdPolicy(seed=seed),
            ],
        ),
        CompetitionScenario(
            name="week_3_mixed_population",
            profile_type="spike",
            seed=seed,
            policy_factories=[
                FollowDemandPolicy,
                LegalInferencePolicy,
                lambda seed=seed: NoisyThresholdPolicy(seed=seed + 1),
                lambda: OscillatingPolicy(phase=1),
            ],
        ),
        CompetitionScenario(
            name="week_4_chaotic_houses",
            profile_type="dspike",
            seed=seed,
            policy_factories=[
                PriceAwarePolicy,
                lambda seed=seed: NoisyThresholdPolicy(seed=seed + 2, noise_scale=0.08),
                lambda: OscillatingPolicy(period=3),
                VolatilitySeekingPolicy,
            ],
        ),
    ]


def scenario_names(seed: int = 1) -> list[str]:
    """Return the built-in weekly scenario IDs in registry order."""
    return [scenario.name for scenario in weekly_training_scenarios(seed=seed)]


def format_scenario_choices(seed: int = 1) -> str:
    """Return a newline-delimited scenario list for CLI discovery."""
    return "\n".join(scenario_names(seed=seed))


def scenario_by_name(
    name: str,
    scenarios: list[CompetitionScenario] | None = None,
) -> CompetitionScenario:
    scenarios_by_name = {
        scenario.name: scenario
        for scenario in (scenarios if scenarios is not None else weekly_training_scenarios())
    }
    try:
        return scenarios_by_name[name]
    except KeyError as exc:
        choices = ", ".join(scenarios_by_name)
        raise ValueError(f"unknown scenario {name!r}; choices: {choices}") from exc


def _validate_unique_policy_names(
    scenario_name: str,
    policies: list[HousePolicy],
) -> None:
    names = [policy.name for policy in policies]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"scenario {scenario_name!r} has duplicate policy name(s): "
            f"{', '.join(duplicates)}"
        )
