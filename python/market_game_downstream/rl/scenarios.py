"""Competition scenario helpers for downstream market-game experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import random
from typing import Any

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

SCENARIO_CONFIG_PATH = Path(__file__).with_name("scenario_configs").joinpath("weekly.json")

POLICY_TYPES: dict[str, type[HousePolicy]] = {
    "FlattenDemandPolicy": FlattenDemandPolicy,
    "FollowDemandPolicy": FollowDemandPolicy,
    "FullCyclePolicy": FullCyclePolicy,
    "LegalInferencePolicy": LegalInferencePolicy,
    "NoisyThresholdPolicy": NoisyThresholdPolicy,
    "OscillatingPolicy": OscillatingPolicy,
    "PriceAwarePolicy": PriceAwarePolicy,
    "RollingPricePolicy": RollingPricePolicy,
    "VolatilitySeekingPolicy": VolatilitySeekingPolicy,
}


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

    def to_env_config(self) -> tuple[list[HousePolicy], MarketGameConfig]:
        """Return opponent policies and rule config for Gym-style training."""
        return (
            self.policies(),
            replace(self.config, demand_profile=self.demand()),
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
    return load_scenarios(seed=seed)


def load_scenarios(
    path: str | Path | None = SCENARIO_CONFIG_PATH,
    seed: int | None = None,
) -> list[CompetitionScenario]:
    """Load scenario definitions from a JSON config file.

    The file-level seed is used when a scenario omits one. The caller-provided
    seed overrides both, which makes repeated train/evaluation runs easy to
    sweep without editing config files.
    """
    config_path = Path(path) if path is not None else SCENARIO_CONFIG_PATH
    data = json.loads(config_path.read_text(encoding="utf-8"))
    default_seed = int(data.get("seed", 1))
    scenarios = data.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("scenario config must contain a 'scenarios' list")
    return [
        _scenario_from_config(item, default_seed=default_seed, override_seed=seed)
        for item in scenarios
    ]


def legacy_weekly_training_scenarios(seed: int = 1) -> list[CompetitionScenario]:
    """Return the original hard-coded weekly curriculum.

    Kept as an escape hatch for tests and comparisons; production training and
    evaluation should use ``weekly_training_scenarios``/``load_scenarios``.
    """
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


def scenario_by_name(
    name: str,
    scenarios: list[CompetitionScenario] | None = None,
) -> CompetitionScenario:
    """Return one scenario by name with a clear error for CLI callers."""
    scenarios = scenarios if scenarios is not None else weekly_training_scenarios()
    scenarios_by_name = {scenario.name: scenario for scenario in scenarios}
    try:
        return scenarios_by_name[name]
    except KeyError as exc:
        choices = ", ".join(scenarios_by_name)
        raise ValueError(f"unknown scenario {name!r}; choices: {choices}") from exc


def _scenario_from_config(
    item: object,
    default_seed: int,
    override_seed: int | None,
) -> CompetitionScenario:
    if not isinstance(item, dict):
        raise ValueError("each scenario entry must be an object")
    seed = int(override_seed if override_seed is not None else item.get("seed", default_seed))
    return CompetitionScenario(
        name=_required_string(item, "name"),
        profile_type=str(item.get("profile_type", "profile1")),
        seed=seed,
        policy_factories=_policy_factories(item.get("opponents", []), scenario_seed=seed),
    )


def _policy_factories(items: object, scenario_seed: int) -> list[PolicyFactory]:
    if not isinstance(items, list):
        raise ValueError("scenario 'opponents' must be a list")
    house_count = 1 + _opponent_count(items)
    factories: list[PolicyFactory] = []
    next_index = 0
    for item in items:
        group_factories = _policy_factories_from_item(
            item,
            scenario_seed=scenario_seed,
            house_count=house_count,
            start_index=next_index,
        )
        next_index += len(group_factories)
        factories.extend(group_factories)
    return factories


def _opponent_count(items: list[object]) -> int:
    total = 0
    for item in items:
        if isinstance(item, str):
            total += 1
        elif isinstance(item, dict):
            total += int(item.get("count", 1))
        else:
            raise ValueError("opponent entries must be policy names or objects")
    return total


def _policy_factories_from_item(
    item: object,
    scenario_seed: int,
    house_count: int,
    start_index: int,
) -> list[PolicyFactory]:
    if isinstance(item, str):
        return [
            _policy_factory(
                item,
                scenario_seed=scenario_seed,
                copy_index=start_index,
                local_index=0,
                house_count=house_count,
            )
        ]
    if not isinstance(item, dict):
        raise ValueError("opponent entries must be policy names or objects")
    if "grab_bag" in item:
        return _grab_bag_factories(
            item,
            scenario_seed=scenario_seed,
            house_count=house_count,
            start_index=start_index,
        )

    count = int(item.get("count", 1))
    if count < 1:
        raise ValueError("opponent count must be at least 1")
    return [
        _policy_factory(
            item,
            scenario_seed=scenario_seed,
            copy_index=start_index + index,
            local_index=index,
            house_count=house_count,
        )
        for index in range(count)
    ]


def _grab_bag_factories(
    item: dict[str, object],
    scenario_seed: int,
    house_count: int,
    start_index: int,
) -> list[PolicyFactory]:
    count = int(item.get("count", 1))
    if count < 1:
        raise ValueError("grab_bag count must be at least 1")
    choices = item.get("grab_bag")
    if not isinstance(choices, list) or not choices:
        raise ValueError("grab_bag must be a non-empty list")

    rng_seed = _resolve_placeholders(
        item.get("seed", "$seed"),
        scenario_seed=scenario_seed,
        copy_index=start_index,
        local_index=0,
        house_count=house_count,
        policy_type="",
    )
    rng = random.Random(int(rng_seed))
    weights = [_choice_weight(choice) for choice in choices]
    name_template = str(item.get("name_template", "$type_$index"))
    factories = []
    for local_index in range(count):
        choice = rng.choices(choices, weights=weights, k=1)[0]
        policy_item = _normalize_grab_bag_choice(choice)
        kwargs = dict(policy_item.get("kwargs", {}))
        policy_type = _required_string(policy_item, "type")
        kwargs.setdefault("name", name_template)
        concrete_item = {
            "type": policy_type,
            "kwargs": kwargs,
        }
        factories.append(
            _policy_factory(
                concrete_item,
                scenario_seed=scenario_seed,
                copy_index=start_index + local_index,
                local_index=local_index,
                house_count=house_count,
            )
        )
    return factories


def _choice_weight(choice: object) -> float:
    if isinstance(choice, str):
        return 1.0
    if isinstance(choice, dict):
        return float(choice.get("weight", 1.0))
    raise ValueError("grab_bag choices must be policy names or objects")


def _normalize_grab_bag_choice(choice: object) -> dict[str, object]:
    if isinstance(choice, str):
        return {"type": choice, "kwargs": {}}
    if isinstance(choice, dict):
        return {
            "type": _required_string(choice, "type"),
            "kwargs": dict(choice.get("kwargs", {})),
        }
    raise ValueError("grab_bag choices must be policy names or objects")


def _policy_factory_with_index(
    item: object,
    scenario_seed: int,
    copy_index: int,
    local_index: int,
    house_count: int,
) -> PolicyFactory:
    if isinstance(item, str):
        policy_type = item
        kwargs: dict[str, Any] = {}
    elif isinstance(item, dict):
        policy_type = _required_string(item, "type")
        kwargs = dict(item.get("kwargs", {}))
    else:
        raise ValueError("opponent entries must be policy names or objects")

    policy_class = POLICY_TYPES.get(policy_type)
    if policy_class is None:
        choices = ", ".join(sorted(POLICY_TYPES))
        raise ValueError(f"unknown policy type {policy_type!r}; choices: {choices}")
    kwargs = _resolve_placeholders(
        kwargs,
        scenario_seed=scenario_seed,
        copy_index=copy_index,
        local_index=local_index,
        house_count=house_count,
        policy_type=policy_type,
    )
    return lambda policy_class=policy_class, kwargs=kwargs: policy_class(**kwargs)


def _policy_factory(
    item: object,
    scenario_seed: int,
    copy_index: int = 0,
    local_index: int = 0,
    house_count: int = 1,
) -> PolicyFactory:
    return _policy_factory_with_index(
        item,
        scenario_seed=scenario_seed,
        copy_index=copy_index,
        local_index=local_index,
        house_count=house_count,
    )


def _resolve_placeholders(
    value: object,
    scenario_seed: int,
    copy_index: int,
    local_index: int,
    house_count: int,
    policy_type: str,
) -> object:
    if isinstance(value, str) and value == "$seed":
        return scenario_seed
    if isinstance(value, str) and value == "$index":
        return copy_index
    if isinstance(value, str) and value == "$local_index":
        return local_index
    if isinstance(value, str) and value == "$house_count":
        return house_count
    if isinstance(value, str) and value == "$seed+$index":
        return scenario_seed + copy_index
    if isinstance(value, str) and value == "$seed-$index":
        return scenario_seed - copy_index
    if isinstance(value, str) and value == "$seed+$local_index":
        return scenario_seed + local_index
    if isinstance(value, str) and value == "$seed-$local_index":
        return scenario_seed - local_index
    if isinstance(value, str) and value.startswith("$seed+"):
        return scenario_seed + int(value.removeprefix("$seed+"))
    if isinstance(value, str) and value.startswith("$seed-"):
        return scenario_seed - int(value.removeprefix("$seed-"))
    if isinstance(value, str):
        return (
            value.replace("$local_index", str(local_index))
            .replace("$index", str(copy_index))
            .replace("$house_count", str(house_count))
            .replace("$type", policy_type.removesuffix("Policy"))
            .replace("$seed", str(scenario_seed))
        )
    if isinstance(value, dict):
        return {
            key: _resolve_placeholders(
                item,
                scenario_seed=scenario_seed,
                copy_index=copy_index,
                local_index=local_index,
                house_count=house_count,
                policy_type=policy_type,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_placeholders(
                item,
                scenario_seed=scenario_seed,
                copy_index=copy_index,
                local_index=local_index,
                house_count=house_count,
                policy_type=policy_type,
            )
            for item in value
        ]
    return value


def _required_string(item: dict[str, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"scenario config entry must define non-empty string {key!r}")
    return value


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
