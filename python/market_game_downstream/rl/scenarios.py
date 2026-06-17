"""Built-in RL scenarios and CSV-friendly evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import random
from pathlib import Path

from python.market_game_downstream.core import (
    DEFAULT_CONFIG,
    HourRecord,
    HousePolicy,
    MarketGameConfig,
    MarketScenario,
    demand_profile,
    run_scenario,
)
from python.market_game_downstream.rl.metrics import result_price_volatility
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


def submitted_function_policy_factory(
    path: str | Path,
    name: str = "SubmittedHouse",
) -> PolicyFactory:
    if not isinstance(name, str) or not name:
        raise ValueError("submitted function policy name must be a non-empty string")
    submission_path = Path(path)

    def factory(path: Path = submission_path, name: str = name) -> HousePolicy:
        from python.market_game_downstream.rl.export.export_policy import (
            FunctionSubmissionPolicy,
        )
        from python.market_game_downstream.rl.export.validators import load_compute_demand

        return FunctionSubmissionPolicy(load_compute_demand(path), name=name)

    return factory


def evaluate_scenario(scenario: CompetitionScenario) -> list[dict[str, str]]:
    result = run_scenario(scenario.to_market_scenario())
    volatility = result_price_volatility(result)
    return [
        _summary_row(scenario, house, result.records, volatility)
        for house in result.houses
    ]


def evaluate_curriculum(seed: int = 1) -> list[dict[str, str]]:
    return [
        row
        for scenario in weekly_training_scenarios(seed=seed)
        for row in evaluate_scenario(scenario)
    ]


def _summary_row(
    scenario: CompetitionScenario,
    house: object,
    records: list[HourRecord],
    price_volatility: float,
) -> dict[str, str]:
    penalty_cost = sum(
        record.penalties_by_house.get(house.policy.name, 0.0)
        for record in records
    )
    invalid_load_adjustment = sum(
        record.invalid_load_adjustments_by_house.get(house.policy.name, 0.0)
        for record in records
    )
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
        "invalid_load_adjustment": f"{invalid_load_adjustment:.10f}",
        "penalty_cost": f"{penalty_cost:.10f}",
        "price_volatility": f"{price_volatility:.10f}",
    }


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
