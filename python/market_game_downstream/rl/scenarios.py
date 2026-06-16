"""Competition scenario helpers for downstream market-game experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import random
from typing import Any

from python.market_game_downstream.core import (
    DEFAULT_CONFIG,
    MarketGameConfig,
    MarketScenario,
    PROFILE_TYPES,
    demand_profile,
)
from python.market_game_downstream.core.simulator import HourRecord, HousePolicy, run_scenario
from python.market_game_downstream.rl.core.metrics import result_price_volatility
from .agents.policies import (
    FlattenDemandPolicy,
    FollowDemandPolicy,
    FullCyclePolicy,
    InvalidDemandPolicy,
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
    "InvalidDemandPolicy": InvalidDemandPolicy,
    "LegalInferencePolicy": LegalInferencePolicy,
    "NoisyThresholdPolicy": NoisyThresholdPolicy,
    "OscillatingPolicy": OscillatingPolicy,
    "PriceAwarePolicy": PriceAwarePolicy,
    "RollingPricePolicy": RollingPricePolicy,
    "VolatilitySeekingPolicy": VolatilitySeekingPolicy,
}

SUBMISSION_POLICY_TYPES = {
    "FunctionSubmissionPolicy",
    "SubmittedFunctionPolicy",
}


@dataclass(frozen=True)
class CompetitionScenario:
    """Configurable weekly scenario for local training and evaluation."""

    name: str
    profile_type: str = "profile1"
    seed: int = 1
    policy_factories: list[PolicyFactory] = field(default_factory=list)
    players: list[dict[str, Any]] = field(default_factory=list)
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
        """Return opponent policies and rule config for Gym-style training."""
        return (
            self.policies(),
            replace(self.config, demand_profile=self.demand()),
        )

    def training_observation_mode(self) -> str | None:
        """Return scenario-declared learner observation mode, when present."""
        for player in self.players:
            if player.get("role") == "rl_training_agent":
                mode = player.get("observation_mode")
                if isinstance(mode, str) and mode:
                    return mode
        return None

    def submitted_players(self) -> list[dict[str, Any]]:
        """Return submitted-function player metadata for validation/HELICS."""
        return [
            dict(player)
            for player in self.players
            if player.get("role") == "submitted_function"
        ]

    def with_policy_factory(
        self,
        policy_factory: PolicyFactory,
        *,
        first: bool = True,
    ) -> "CompetitionScenario":
        """Return a scenario copy with one additional executable policy."""
        if first:
            policy_factories = [policy_factory, *self.policy_factories]
        else:
            policy_factories = [*self.policy_factories, policy_factory]
        return replace(self, policy_factories=policy_factories)


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
    if not isinstance(data, dict):
        raise ValueError("scenario config must be a JSON object")
    default_seed = _int_value(data.get("seed", 1), "scenario config seed")
    scenarios = data.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("scenario config must contain a 'scenarios' list")
    loaded = [
        _scenario_from_config(item, default_seed=default_seed, override_seed=seed)
        for item in scenarios
    ]
    _validate_unique_names(loaded)
    return loaded


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


def submitted_function_policy_factory(
    path: str | Path,
    name: str = "SubmittedHouse",
) -> PolicyFactory:
    """Return a policy factory for a standalone ``compute_demand`` file."""
    if not isinstance(name, str) or not name:
        raise ValueError("submitted function policy name must be a non-empty string")
    submission_path = Path(path)

    def factory(
        path: Path = submission_path,
        name: str = name,
    ) -> HousePolicy:
        from python.market_game_downstream.rl.export.export_policy import (
            FunctionSubmissionPolicy,
        )
        from python.market_game_downstream.rl.export.validators import load_compute_demand

        return FunctionSubmissionPolicy(load_compute_demand(path), name=name)

    return factory


def _validate_unique_names(scenarios: list[CompetitionScenario]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for scenario in scenarios:
        if scenario.name in seen:
            duplicates.append(scenario.name)
        seen.add(scenario.name)
    if duplicates:
        names = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"duplicate scenario name(s): {names}")


def _validate_unique_policy_names(
    scenario_name: str,
    policies: list[HousePolicy],
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for policy in policies:
        if policy.name in seen:
            duplicates.append(policy.name)
        seen.add(policy.name)
    if duplicates:
        names = ", ".join(sorted(set(duplicates)))
        raise ValueError(
            f"scenario {scenario_name!r} has duplicate policy name(s): {names}"
        )


def _scenario_from_config(
    item: object,
    default_seed: int,
    override_seed: int | None,
) -> CompetitionScenario:
    if not isinstance(item, dict):
        raise ValueError("each scenario entry must be an object")
    seed = _int_value(
        override_seed if override_seed is not None else item.get("seed", default_seed),
        "scenario seed",
    )
    return CompetitionScenario(
        name=_required_string(item, "name"),
        profile_type=_profile_type(item.get("profile_type", "profile1")),
        seed=seed,
        policy_factories=_scenario_policy_factories(item, scenario_seed=seed),
        players=_players(item.get("players", [])),
    )


def _scenario_policy_factories(
    item: dict[str, object],
    scenario_seed: int,
) -> list[PolicyFactory]:
    opponent_entries = item.get("opponents", [])
    player_entries = _submitted_player_opponent_entries(item.get("players", []))
    mirrored_keys = _submitted_entry_keys(opponent_entries)
    extra_player_entries = [
        entry for entry in player_entries if _submitted_entry_key(entry) not in mirrored_keys
    ]
    return _policy_factories(
        [*(_opponent_entries(opponent_entries)), *extra_player_entries],
        scenario_seed=scenario_seed,
    )


def _policy_factories(items: object, scenario_seed: int) -> list[PolicyFactory]:
    items = _opponent_entries(items)
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
            total += _positive_int(item.get("count", 1), "opponent count")
        else:
            raise ValueError("opponent entries must be policy names or objects")
    return total


def _opponent_entries(items: object) -> list[object]:
    if not isinstance(items, list):
        raise ValueError("scenario 'opponents' must be a list")
    return list(items)


def _players(items: object) -> list[dict[str, Any]]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("scenario 'players' must be a list")
    players: list[dict[str, Any]] = []
    for player in items:
        if not isinstance(player, dict):
            raise ValueError("scenario player entries must be objects")
        role = player.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("scenario player entries must define non-empty string 'role'")
        players.append(dict(player))
    return players


def _submitted_player_opponent_entries(players: object) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for player in _players(players):
        if player.get("role") != "submitted_function":
            continue
        path = player.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("submitted_function players must define non-empty string 'path'")
        name = player.get("name", "SubmittedHouse")
        if not isinstance(name, str) or not name:
            raise ValueError("submitted_function player name must be a non-empty string")
        count = _positive_int(player.get("count", 1), "submitted_function count")
        entry: dict[str, object] = {
            "type": "SubmittedFunctionPolicy",
            "path": path,
            "kwargs": {"name": name},
        }
        if count != 1:
            entry["count"] = count
        entries.append(entry)
    return entries


def _submitted_entry_keys(items: object) -> set[tuple[str, str]]:
    return {
        key
        for item in _opponent_entries(items)
        if (key := _submitted_entry_key(item)) is not None
    }


def _submitted_entry_key(item: object) -> tuple[str, str] | None:
    if not isinstance(item, dict):
        return None
    if item.get("type") not in SUBMISSION_POLICY_TYPES:
        return None
    path = item.get("path")
    kwargs = item.get("kwargs", {})
    name = kwargs.get("name", "SubmittedHouse") if isinstance(kwargs, dict) else "SubmittedHouse"
    if isinstance(path, str) and isinstance(name, str):
        return (path, name)
    return None


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

    count = _positive_int(item.get("count", 1), "opponent count")
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
    count = _positive_int(item.get("count", 1), "grab_bag count")
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
    rng = random.Random(_int_value(rng_seed, "grab_bag seed"))
    weights = [_choice_weight(choice) for choice in choices]
    if sum(weights) <= 0.0:
        raise ValueError("grab_bag weights must contain at least one positive value")
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
        weight = _float_value(choice.get("weight", 1.0), "grab_bag weight")
        if weight < 0.0:
            raise ValueError("grab_bag weight must be non-negative")
        return weight
    raise ValueError("grab_bag choices must be policy names or objects")


def _normalize_grab_bag_choice(choice: object) -> dict[str, object]:
    if isinstance(choice, str):
        return {"type": choice, "kwargs": {}}
    if isinstance(choice, dict):
        return {
            "type": _required_string(choice, "type"),
            "kwargs": _kwargs_dict(choice.get("kwargs", {})),
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
        kwargs = _kwargs_dict(item.get("kwargs", {}))
    else:
        raise ValueError("opponent entries must be policy names or objects")

    if policy_type in SUBMISSION_POLICY_TYPES:
        return _submitted_function_factory(
            item,
            kwargs=kwargs,
            scenario_seed=scenario_seed,
            copy_index=copy_index,
            local_index=local_index,
            house_count=house_count,
            policy_type=policy_type,
        )

    policy_class = POLICY_TYPES.get(policy_type)
    if policy_class is None:
        choices = ", ".join(sorted([*POLICY_TYPES, *SUBMISSION_POLICY_TYPES]))
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


def _submitted_function_factory(
    item: object,
    kwargs: dict[str, Any],
    scenario_seed: int,
    copy_index: int,
    local_index: int,
    house_count: int,
    policy_type: str,
) -> PolicyFactory:
    if not isinstance(item, dict):
        raise ValueError("submitted function policy entries must be objects")
    path = item.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "submitted function policy entries must define non-empty string 'path'"
        )
    resolved_path = _resolve_placeholders(
        path,
        scenario_seed=scenario_seed,
        copy_index=copy_index,
        local_index=local_index,
        house_count=house_count,
        policy_type=policy_type,
    )
    resolved_kwargs = _resolve_placeholders(
        kwargs,
        scenario_seed=scenario_seed,
        copy_index=copy_index,
        local_index=local_index,
        house_count=house_count,
        policy_type=policy_type,
    )
    if not isinstance(resolved_path, str):
        raise ValueError("submitted function policy path must resolve to a string")
    name = resolved_kwargs.pop("name", "SubmittedHouse")
    if resolved_kwargs:
        extra = ", ".join(sorted(resolved_kwargs))
        raise ValueError(f"unsupported submitted function policy kwargs: {extra}")
    return submitted_function_policy_factory(resolved_path, name=name)


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
        suffix = value.removeprefix("$seed+")
        if not suffix.lstrip("-").isdigit():
            raise ValueError(f"invalid seed placeholder {value!r}")
        return scenario_seed + int(suffix)
    if isinstance(value, str) and value.startswith("$seed-"):
        suffix = value.removeprefix("$seed-")
        if not suffix.lstrip("-").isdigit():
            raise ValueError(f"invalid seed placeholder {value!r}")
        return scenario_seed - int(suffix)
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


def _profile_type(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("scenario profile_type must be a non-empty string")
    if value not in PROFILE_TYPES:
        choices = ", ".join(PROFILE_TYPES)
        raise ValueError(f"unknown demand profile {value!r}; choices: {choices}")
    return value


def _positive_int(value: object, field_name: str) -> int:
    result = _int_value(value, field_name)
    if result < 1:
        raise ValueError(f"{field_name} must be an integer >= 1")
    return result


def _int_value(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    raise ValueError(f"{field_name} must be an integer")


def _float_value(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _kwargs_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("policy kwargs must be an object")
    return dict(value)


def evaluate_scenario(scenario: CompetitionScenario) -> list[dict[str, str]]:
    """Run one scenario and return CSV-friendly summary rows."""
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
