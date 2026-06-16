"""Focused checks for downstream scenario orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from python.market_game_downstream.core.simulator import run_scenario
from python.market_game_downstream.rl.aggregate_scenarios import aggregate_rows
from python.market_game_downstream.rl.evaluate_scenarios import _parse_seed_list, rows_for_args
from python.market_game_downstream.rl.evaluate_submission import practice_rows
from python.market_game_downstream.rl.export import SubmissionValidationError
from python.market_game_downstream.rl.scenario_builder import ScenarioConfigBuilder
from python.market_game_downstream.rl.scenarios import (
    evaluate_scenario,
    load_scenarios,
    stock_example_scenario,
    submitted_function_policy_factory,
)


def run_scenario_smoke_check() -> None:
    check_stock_scenario_execution()
    check_config_loading_and_rejection()
    check_held_out_and_invalid_demand_configs()
    check_builder_and_player_metadata()
    check_submission_evaluation_paths()
    check_seed_and_aggregate_helpers()


def check_stock_scenario_execution() -> None:
    scenario = stock_example_scenario()
    result = run_scenario(scenario.to_market_scenario())
    assert len(result.records) == 24
    assert result.oracle_trace()[0].loads_by_house

    rows = evaluate_scenario(scenario)
    assert {row["agent"] for row in rows} == {
        "FlattenDemandHouse",
        "FullCycleHouse",
        "PriceAwareHouse",
    }
    first = rows[0]
    assert {"invalid_load_adjustment", "penalty_cost", "price_volatility"} <= first.keys()
    assert float(first["price_volatility"]) > 0.0


def check_config_loading_and_rejection() -> None:
    loaded = load_scenarios(seed=3)
    assert [scenario.name for scenario in loaded] == [
        "week_1_baselines",
        "week_2_new_profile",
        "week_3_mixed_population",
        "week_4_chaotic_houses",
    ]
    assert loaded[1].policies()[-1].seed == 3
    assert loaded[2].policies()[2].seed == 4

    large_config = (
        Path(__file__).resolve().parents[1]
        / "rl"
        / "scenario_configs"
        / "large_population.json"
    )
    large = load_scenarios(large_config, seed=5)
    assert len(large[1].policies()) == 40
    assert len({policy.name for policy in large[1].policies()}) == 40

    assert_config_error([], "scenario config must be a JSON object")
    assert_config_error(
        {"scenarios": [{"name": "bad", "profile_type": "typo", "opponents": []}]},
        "unknown demand profile",
    )
    assert_config_error(
        {"scenarios": [{"name": "bad", "opponents": ["TypoPolicy"]}]},
        "unknown policy type",
    )
    assert_config_error(
        {
            "scenarios": [
                {
                    "name": "bad",
                    "opponents": [
                        {
                            "type": "NoisyThresholdPolicy",
                            "kwargs": {"seed": "$seed+bad"},
                        }
                    ],
                }
            ]
        },
        "invalid seed placeholder",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "scenarios.json"
        path.write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "name": "duplicate_policy_names",
                            "opponents": [{"type": "PriceAwarePolicy", "count": 2}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        scenario = load_scenarios(path)[0]
        try:
            scenario.policies()
        except ValueError as exc:
            assert "duplicate policy name" in str(exc)
        else:
            raise AssertionError("duplicate policy names were not rejected")


def check_held_out_and_invalid_demand_configs() -> None:
    held_out = load_scenarios(scenario_config_path("held_out_validation.json"), seed=11)
    weekly_names = {scenario.name for scenario in load_scenarios(seed=11)}
    held_out_names = {scenario.name for scenario in held_out}
    assert held_out_names == {
        "validation_profile1_inference_mix",
        "validation_random_grab_bag_24",
        "validation_dspike_volatile_30",
    }
    assert not held_out_names.intersection(weekly_names)
    assert len(held_out[1].policies()) == 24
    assert len(held_out[2].policies()) == 30

    invalid_stress = load_scenarios(
        scenario_config_path("invalid_demand_stress.json"),
        seed=11,
    )[0]
    rows = evaluate_scenario(invalid_stress)
    invalid_row = next(row for row in rows if row["agent"] == "InvalidOvercharger")
    assert int(invalid_row["boundary_warnings"]) == 24
    assert int(invalid_row["clamps"]) == 24
    assert float(invalid_row["invalid_load_adjustment"]) > 0.0
    assert float(invalid_row["penalty_cost"]) > 0.0


def check_builder_and_player_metadata() -> None:
    submission = example_submission_path()
    builder = ScenarioConfigBuilder(seed=7)
    (
        builder.scenario("builder_mixed", profile_type="profile1")
        .training_agent(observation_mode="local")
        .submitted_function(submission, name="BuilderSubmission")
        .strategies(
            "PriceAwarePolicy",
            count=2,
            name_template="BuilderPrice_$local_index",
        )
        .grab_bag(
            count=1,
            choices=[{"type": "NoisyThresholdPolicy", "kwargs": {"seed": "$seed+$index"}}],
        )
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "builder.json"
        builder.write(path)
        scenario = load_scenarios(path, seed=9)[0]
        assert scenario.training_observation_mode() == "local"
        assert scenario.submitted_players()[0]["name"] == "BuilderSubmission"
        assert {policy.name for policy in scenario.policies()} == {
            "BuilderSubmission",
            "BuilderPrice_0",
            "BuilderPrice_1",
            "NoisyThreshold_3",
        }


def check_submission_evaluation_paths() -> None:
    submission = example_submission_path()
    submitted_scenario = stock_example_scenario().with_policy_factory(
        submitted_function_policy_factory(submission, name="SmokeSubmittedHouse")
    )
    assert evaluate_scenario(submitted_scenario)[0]["agent"] == "SmokeSubmittedHouse"

    only_submission_rows = rows_for_args(
        argparse.Namespace(
            submission=submission,
            submission_name="OnlySubmittedHouse",
            only_submission=True,
            stock=False,
            scenario="week_1_baselines",
            seed=3,
            seeds=None,
            config=None,
        )
    )
    assert len(only_submission_rows) == 1
    assert only_submission_rows[0]["agent"] == "OnlySubmittedHouse"

    compact_rows = practice_rows(
        argparse.Namespace(
            submission=submission,
            name="PracticeSubmittedHouse",
            scenario="week_1_baselines",
            seed=3,
            config=None,
            include_baselines=False,
        )
    )
    assert compact_rows[0]["house"] == "PracticeSubmittedHouse"
    assert compact_rows[0]["total_cost"] == only_submission_rows[0]["total_cost"]

    assert_practice_submission_error(
        "def compute_demand(price, hour, battery_charge, demand, price_history):\n"
        "    return -999.0\n",
        "failed 24-hour validation",
    )


def check_seed_and_aggregate_helpers() -> None:
    assert _parse_seed_list("1, 2,3") == [1, 2, 3]
    try:
        _parse_seed_list("1,bad")
    except SystemExit as exc:
        assert "invalid seed" in str(exc)
    else:
        raise AssertionError("invalid seed list was not rejected")

    aggregate = aggregate_rows(
        [
            row
            for seed in (1, 2)
            for scenario in load_scenarios(seed=seed)[:1]
            for row in evaluate_scenario(scenario)
        ]
    )
    assert {row["rank"] for row in aggregate} == {"1", "2", "3"}
    assert "total_cost_stdev" in aggregate[0]


def assert_config_error(config: object, expected: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "scenarios.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        try:
            load_scenarios(path)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"scenario config did not fail with {expected!r}")


def assert_practice_submission_error(source: str, expected: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "bad_submission.py"
        path.write_text(source, encoding="utf-8")
        try:
            practice_rows(
                argparse.Namespace(
                    submission=path,
                    name="BadSubmission",
                    scenario="week_1_baselines",
                    seed=3,
                    config=None,
                    include_baselines=False,
                )
            )
        except SubmissionValidationError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"practice submission did not fail with {expected!r}")


def example_submission_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rl"
        / "export"
        / "example_threshold_submission.py"
    )


def scenario_config_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "rl" / "scenario_configs" / name


if __name__ == "__main__":
    run_scenario_smoke_check()
    print("scenario smoke: ok")
