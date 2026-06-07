"""Smoke checks for downstream competition scenarios."""

from __future__ import annotations

from python.market_game_downstream.core.simulator import run_scenario
from python.market_game_downstream.rl.scenarios import (
    evaluate_curriculum,
    stock_example_scenario,
)


def run_scenario_smoke_check() -> None:
    stock_result = run_scenario(stock_example_scenario().to_market_scenario())
    assert len(stock_result.records) == 24
    assert stock_result.oracle_trace()[0].hour == 0
    assert stock_result.oracle_trace()[0].loads_by_house

    rows = evaluate_curriculum(seed=3)
    assert rows
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
