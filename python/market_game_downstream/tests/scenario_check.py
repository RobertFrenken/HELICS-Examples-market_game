"""Focused checks for retained scenario orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from python.market_game_downstream.core import run_scenario
from python.market_game_downstream.rl.evaluate import (
    _parse_seed_list,
    evaluate_scenario,
    rows_for_args,
)
from python.market_game_downstream.rl.envs.scenarios import (
    CompetitionScenario,
    stock_example_scenario,
    weekly_training_scenarios,
)


def run_scenario_smoke_check() -> None:
    check_stock_scenario_execution()
    check_builtin_scenario_catalog()
    check_submission_evaluation_path()
    check_composed_agent_evaluation_path()
    check_seed_helper()


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


def check_builtin_scenario_catalog() -> None:
    loaded = weekly_training_scenarios(seed=3)
    assert [scenario.name for scenario in loaded] == [
        "week_1_baselines",
        "week_2_new_profile",
        "week_3_mixed_population",
        "week_4_chaotic_houses",
    ]
    assert loaded[1].policies()[-1].name == "NoisyThresholdHouse"
    assert loaded[2].policies()[2].name == "NoisyThresholdHouse"

    duplicate_names = CompetitionScenario(
        name="duplicate_policy_names",
        policy_factories=[loaded[0].policy_factories[1], loaded[0].policy_factories[1]],
    )
    try:
        duplicate_names.policies()
    except ValueError as exc:
        assert "duplicate policy name" in str(exc)
    else:
        raise AssertionError("duplicate policy names were not rejected")


def check_submission_evaluation_path() -> None:
    rows = rows_for_args(
        argparse.Namespace(
            submission=example_submission_path(),
            submission_name="OnlySubmittedHouse",
            only_submission=True,
            agent_config=None,
            agent_name=None,
            stock=False,
            scenario="week_1_baselines",
            seed=3,
            seeds=None,
        )
    )
    assert len(rows) == 1
    assert rows[0]["agent"] == "OnlySubmittedHouse"
    assert rows[0]["clamps"] == "0"


def check_composed_agent_evaluation_path() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "agent.toml"
        config_path.write_text(
            "\n".join(
                [
                    "[agent]",
                    'name = "ConfigRollingHouse"',
                    "",
                    "[agent.controller]",
                    'type = "rolling_threshold"',
                    "reserve = 2.0",
                ]
            ),
            encoding="utf-8",
        )
        rows = rows_for_args(
            argparse.Namespace(
                submission=None,
                submission_name="SubmittedHouse",
                only_submission=True,
                agent_config=config_path,
                agent_name="ComposedRollingHouse",
                stock=False,
                scenario="week_1_baselines",
                seed=3,
                seeds=None,
            )
        )
    assert len(rows) == 1
    assert rows[0]["agent"] == "ComposedRollingHouse"
    assert rows[0]["clamps"] == "0"


def check_seed_helper() -> None:
    assert _parse_seed_list("1, 2,3") == [1, 2, 3]
    try:
        _parse_seed_list("1,bad")
    except SystemExit as exc:
        assert "invalid seed" in str(exc)
    else:
        raise AssertionError("invalid seed list was not rejected")


def example_submission_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rl"
        / "export"
        / "example_threshold_submission.py"
    )


if __name__ == "__main__":
    run_scenario_smoke_check()
    print("scenario smoke: ok")
