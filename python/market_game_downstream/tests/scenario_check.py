"""Smoke checks for downstream competition scenarios."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from python.market_game_downstream.core.simulator import run_scenario
from python.market_game_downstream.rl.scenarios import (
    evaluate_curriculum,
    evaluate_scenario,
    load_scenarios,
    stock_example_scenario,
)
from python.market_game_downstream.rl.evaluate_scenarios import _parse_seed_list


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


def run_scenario_smoke_check() -> None:
    stock_result = run_scenario(stock_example_scenario().to_market_scenario())
    assert len(stock_result.records) == 24
    assert stock_result.oracle_trace()[0].hour == 0
    assert stock_result.oracle_trace()[0].loads_by_house

    rows = evaluate_curriculum(seed=3)
    assert rows
    stock_rows = evaluate_scenario(stock_example_scenario())
    for row in stock_rows:
        assert "invalid_load_adjustment" in row
        assert "penalty_cost" in row
        assert "price_volatility" in row
        assert float(row["invalid_load_adjustment"]) == 0.0
        assert float(row["penalty_cost"]) == 0.0
        assert float(row["price_volatility"]) > 0.0
    loaded = load_scenarios(seed=3)
    assert [scenario.name for scenario in loaded] == [
        "week_1_baselines",
        "week_2_new_profile",
        "week_3_mixed_population",
        "week_4_chaotic_houses",
    ]
    assert loaded[1].seed == 3
    assert loaded[1].policies()[-1].seed == 3
    assert loaded[2].policies()[2].seed == 4
    large_config = (
        Path(__file__).resolve().parents[1]
        / "rl"
        / "scenario_configs"
        / "large_population.json"
    )
    large = load_scenarios(large_config, seed=5)
    assert len(large[0].policies()) == 25
    assert len(large[1].policies()) == 40
    assert len(large[2].policies()) == 50
    noisy = [policy for policy in large[0].policies() if policy.name.startswith("NoisyThreshold_")]
    assert noisy[0].seed == 20
    assert noisy[-1].seed == 24
    grab_bag_names = [policy.name for policy in large[1].policies()]
    assert len(grab_bag_names) == len(set(grab_bag_names))
    assert grab_bag_names == [policy.name for policy in load_scenarios(large_config, seed=5)[1].policies()]
    inference = [policy for policy in large[1].policies() if policy.name.startswith("LegalInference_")]
    for policy in inference:
        assert policy.house_count == 41
    scenario_names = {row["scenario"] for row in rows}
    assert "week_1_baselines" in scenario_names
    assert "week_4_chaotic_houses" in scenario_names
    for row in rows:
        assert row["agent"]
        float(row["total_cost"])
        float(row["final_battery"])

    assert_config_error(
        [],
        "scenario config must be a JSON object",
    )
    assert_config_error(
        {
            "scenarios": [
                {"name": "duplicate", "opponents": []},
                {"name": "duplicate", "opponents": []},
            ]
        },
        "duplicate scenario name",
    )
    assert_config_error(
        {
            "scenarios": [
                {
                    "name": "bad_count",
                    "opponents": [{"type": "PriceAwarePolicy", "count": 0}],
                }
            ]
        },
        "opponent count",
    )
    assert_config_error(
        {
            "scenarios": [
                {
                    "name": "bad_profile",
                    "profile_type": "typo",
                    "opponents": [],
                }
            ]
        },
        "unknown demand profile",
    )
    assert_config_error(
        {
            "scenarios": [
                {
                    "name": "unknown_policy",
                    "opponents": ["TypoPolicy"],
                }
            ]
        },
        "unknown policy type",
    )
    assert_config_error(
        {
            "scenarios": [
                {
                    "name": "bad_kwargs",
                    "opponents": [{"type": "PriceAwarePolicy", "kwargs": []}],
                }
            ]
        },
        "policy kwargs",
    )
    assert_config_error(
        {
            "scenarios": [
                {
                    "name": "bad_weight",
                    "opponents": [
                        {
                            "count": 1,
                            "grab_bag": [{"type": "PriceAwarePolicy", "weight": -1}],
                        }
                    ],
                }
            ]
        },
        "grab_bag weight",
    )
    assert_config_error(
        {
            "scenarios": [
                {
                    "name": "bad_placeholder",
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
                            "opponents": [
                                {"type": "PriceAwarePolicy", "count": 2},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        duplicate_name_scenario = load_scenarios(path)[0]
        try:
            duplicate_name_scenario.policies()
        except ValueError as exc:
            assert "duplicate policy name" in str(exc)
        else:
            raise AssertionError("duplicate policy names were not rejected")

    assert _parse_seed_list("1, 2,3") == [1, 2, 3]
    try:
        _parse_seed_list("1,bad")
    except SystemExit as exc:
        assert "invalid seed" in str(exc)
    else:
        raise AssertionError("invalid seed list was not rejected")


if __name__ == "__main__":
    run_scenario_smoke_check()
    print("scenario smoke: ok")
