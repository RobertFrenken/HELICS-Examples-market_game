"""Smoke checks for downstream competition scenarios."""

from __future__ import annotations

from pathlib import Path

from python.market_game_downstream.core.simulator import run_scenario
from python.market_game_downstream.rl.scenarios import (
    evaluate_curriculum,
    load_scenarios,
    stock_example_scenario,
)


def run_scenario_smoke_check() -> None:
    stock_result = run_scenario(stock_example_scenario().to_market_scenario())
    assert len(stock_result.records) == 24
    assert stock_result.oracle_trace()[0].hour == 0
    assert stock_result.oracle_trace()[0].loads_by_house

    rows = evaluate_curriculum(seed=3)
    assert rows
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


if __name__ == "__main__":
    run_scenario_smoke_check()
    print("scenario smoke: ok")
