"""User-facing builder for downstream market-game scenario configs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from python.market_game_downstream.core import PROFILE_TYPES
from python.market_game_downstream.rl.scenarios import POLICY_TYPES


StrategyEntry = str | dict[str, object]


@dataclass
class ScenarioConfigBuilder:
    """Build a JSON-serializable scenario config."""

    seed: int = 1
    scenarios: list["ScenarioBuilder"] = field(default_factory=list)

    def scenario(
        self,
        name: str,
        profile_type: str = "profile1",
        seed: int | None = None,
    ) -> "ScenarioBuilder":
        scenario = ScenarioBuilder(name=name, profile_type=profile_type, seed=seed)
        self.scenarios.append(scenario)
        return scenario

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent) + "\n"

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(self.to_json(), encoding="utf-8")
        return output


@dataclass
class ScenarioBuilder:
    """Build one scenario with explicit player roles and executable opponents."""

    name: str
    profile_type: str = "profile1"
    seed: int | None = None
    opponents: list[StrategyEntry] = field(default_factory=list)
    players: list[dict[str, object]] = field(default_factory=list)

    def demand_profile(self, profile_type: str) -> "ScenarioBuilder":
        _validate_profile_type(profile_type)
        self.profile_type = profile_type
        return self

    def training_agent(
        self,
        name: str = "TrainingAgent",
        observation_mode: str = "price_history",
    ) -> "ScenarioBuilder":
        self.players.append(
            {
                "role": "rl_training_agent",
                "name": name,
                "observation_mode": observation_mode,
            }
        )
        return self

    def strategy(
        self,
        policy_type: str,
        count: int = 1,
        name: str | None = None,
        **kwargs: object,
    ) -> "ScenarioBuilder":
        _validate_policy_type(policy_type)
        _validate_count(count)
        policy_kwargs = dict(kwargs)
        if name is not None:
            policy_kwargs["name"] = name
        self.players.append(
            {
                "role": "strategy",
                "type": policy_type,
                "count": count,
                "kwargs": policy_kwargs,
            }
        )
        self.opponents.append(_opponent_entry(policy_type, count, policy_kwargs))
        return self

    def strategies(
        self,
        policy_type: str,
        count: int,
        name_template: str | None = None,
        **kwargs: object,
    ) -> "ScenarioBuilder":
        if name_template is not None:
            kwargs["name"] = name_template
        return self.strategy(policy_type, count=count, **kwargs)

    def submitted_function(
        self,
        path: str | Path,
        name: str = "SubmittedHouse",
        count: int = 1,
    ) -> "ScenarioBuilder":
        _validate_count(count)
        path_text = Path(path).as_posix()
        policy_name = (
            f"{name}_$local_index"
            if count != 1 and "$" not in name
            else name
        )
        self.players.append(
            {
                "role": "submitted_function",
                "path": path_text,
                "name": policy_name,
                "count": count,
            }
        )
        entry: dict[str, object] = {
            "type": "SubmittedFunctionPolicy",
            "path": path_text,
            "kwargs": {"name": policy_name},
        }
        if count != 1:
            entry["count"] = count
        self.opponents.append(entry)
        return self

    def loaded_rl_agent(
        self,
        checkpoint: str | Path,
        name: str = "LoadedRLAgent",
        observation_mode: str = "price_history",
    ) -> "ScenarioBuilder":
        self.players.append(
            {
                "role": "loaded_rl_agent",
                "name": name,
                "checkpoint": Path(checkpoint).as_posix(),
                "observation_mode": observation_mode,
            }
        )
        return self

    def grab_bag(
        self,
        count: int,
        choices: list[str | dict[str, object]],
        seed: str | int = "$seed",
        name_template: str = "$type_$index",
    ) -> "ScenarioBuilder":
        _validate_count(count)
        if not choices:
            raise ValueError("grab_bag choices must not be empty")
        normalized_choices = [_grab_bag_choice(choice) for choice in choices]
        self.players.append(
            {
                "role": "strategy_grab_bag",
                "count": count,
                "seed": seed,
                "name_template": name_template,
                "choices": normalized_choices,
            }
        )
        self.opponents.append(
            {
                "count": count,
                "seed": seed,
                "name_template": name_template,
                "grab_bag": normalized_choices,
            }
        )
        return self

    def to_dict(self) -> dict[str, object]:
        _validate_profile_type(self.profile_type)
        data: dict[str, object] = {
            "name": self.name,
            "profile_type": self.profile_type,
            "players": self.players,
            "opponents": self.opponents,
        }
        if self.seed is not None:
            data["seed"] = self.seed
        return data


def example_builder() -> ScenarioConfigBuilder:
    """Return a small mixed-population example users can copy and edit."""
    config = ScenarioConfigBuilder(seed=11)
    (
        config.scenario("custom_profile1_mixed", profile_type="profile1")
        .training_agent(observation_mode="price_history")
        .strategies(
            "PriceAwarePolicy",
            count=4,
            name_template="PriceAware_$local_index",
        )
        .strategies(
            "RollingPricePolicy",
            count=4,
            name_template="Rolling_$local_index",
        )
        .grab_bag(
            count=8,
            seed="$seed+100",
            name_template="$type_$index",
            choices=[
                {
                    "type": "NoisyThresholdPolicy",
                    "weight": 3,
                    "kwargs": {"seed": "$seed+$index"},
                },
                {"type": "VolatilitySeekingPolicy", "weight": 1},
                {
                    "type": "LegalInferencePolicy",
                    "weight": 1,
                    "kwargs": {"house_count": "$house_count"},
                },
            ],
        )
    )
    return config


def _opponent_entry(
    policy_type: str,
    count: int,
    kwargs: dict[str, object],
) -> StrategyEntry:
    if count == 1 and not kwargs:
        return policy_type
    entry: dict[str, object] = {"type": policy_type}
    if count != 1:
        entry["count"] = count
    if kwargs:
        entry["kwargs"] = kwargs
    return entry


def _grab_bag_choice(choice: str | dict[str, object]) -> str | dict[str, object]:
    if isinstance(choice, str):
        _validate_policy_type(choice)
        return choice
    if not isinstance(choice, dict):
        raise ValueError("grab_bag choices must be policy names or objects")
    policy_type = choice.get("type")
    if not isinstance(policy_type, str):
        raise ValueError("grab_bag choice must define a policy type")
    _validate_policy_type(policy_type)
    return dict(choice)


def _validate_policy_type(policy_type: str) -> None:
    if policy_type not in POLICY_TYPES:
        choices = ", ".join(sorted(POLICY_TYPES))
        raise ValueError(f"unknown policy type {policy_type!r}; choices: {choices}")


def _validate_profile_type(profile_type: str) -> None:
    if profile_type not in PROFILE_TYPES:
        choices = ", ".join(PROFILE_TYPES)
        raise ValueError(f"unknown demand profile {profile_type!r}; choices: {choices}")


def _validate_count(count: int) -> None:
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be an integer >= 1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "write an example downstream scenario config built with "
            "ScenarioConfigBuilder"
        )
    )
    parser.add_argument(
        "--output",
        help="path to write; prints JSON to stdout when omitted",
    )
    args = parser.parse_args()
    builder = example_builder()
    if args.output:
        builder.write(args.output)
    else:
        print(builder.to_json(), end="")


if __name__ == "__main__":
    main()
