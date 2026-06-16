"""Checks for shared market-game core compatibility."""

from __future__ import annotations

import importlib
import math
from pathlib import Path

from python.market_game_downstream.core import (
    battery_delta_to_market_load,
    normalized_delta_to_market_load,
)
from python.market_game_downstream.core.config import MarketGameConfig, demand_profile
from python.market_game_downstream.core.rules import check_valid, compute_price_from_total_load, ensure_valid
from python.market_game_downstream.core.simulator import (
    BatteryState,
    HouseHourInput,
    MarketScenario,
    run_scenario,
    step_market_hour,
)


def run_shared_core_check() -> None:
    assert compute_price_from_total_load(15.0, 3) == 0.16
    assert demand_profile("profile1")[0:4] == [2, 1, 1, 1]
    assert demand_profile("flat") == [5] * 24
    assert_value_error(lambda: demand_profile("typo"), "unknown demand profile")

    assert_value_error(
        lambda: MarketGameConfig(episode_hours=25, demand_profile=[1.0] * 24),
        "demand_profile",
    )
    assert_value_error(lambda: compute_price_from_total_load(math.nan, 1), "total_market_load")

    battery = BatteryState(0.0)
    assert_value_error(lambda: battery.change(math.nan), "battery delta")
    warning = check_valid(-1.0, 2.0, battery)
    assert warning == "listed consumption exceeds available battery energy"
    assert ensure_valid(-1.0, 2.0, battery) == 2.0

    assert battery_delta_to_market_load(3.0, 2.0, 0.0) == 5.0
    assert battery_delta_to_market_load(3.25, 2.0, 0.0) == 5.25
    assert battery_delta_to_market_load(100.0, 2.0, 19.0) == 3.0
    assert normalized_delta_to_market_load(0.5, 2.0, 0.0) == 4.5
    assert normalized_delta_to_market_load(-0.5, 12.0, 4.0) == 10.0
    assert normalized_delta_to_market_load(2.0, 2.0, 19.0) == 3.0

    scenario = MarketScenario(policies=[])
    assert scenario.config.episode_hours == 24

    record, results = one_house_hour("test", proposed=-1.0, base=2.0, battery=BatteryState(0.0))
    assert results[0].proposed_market_load == -1.0
    assert results[0].market_load == 2.0
    assert record.proposed_loads_by_house["test"] == -1.0
    assert record.loads_by_house["test"] == 2.0
    assert results[0].energy_cost == 1.0
    assert results[0].penalty_cost == 60.0
    assert results[0].invalid_load_adjustment == 3.0
    assert results[0].cost == 61.0
    assert record.energy_costs_by_house["test"] == 1.0
    assert record.penalties_by_house["test"] == 60.0
    assert record.invalid_load_adjustments_by_house["test"] == 3.0
    assert record.warnings_by_house["test"] == "listed consumption exceeds available battery energy"

    battery = BatteryState(0.0)
    _, results = one_house_hour("over_charge_rate", proposed=100.0, base=2.0, battery=battery)
    assert results[0].market_load == 7.0
    assert battery.energy == 5.0
    assert results[0].warning == "listed battery charge rate exceeds maximum charge rate"

    battery = BatteryState(19.0)
    _, results = one_house_hour("over_capacity", proposed=100.0, base=2.0, battery=battery)
    assert results[0].market_load == 3.0
    assert battery.energy == 20.0
    assert results[0].warning == "listed battery charge rate exceeds available battery storage capacity"

    battery = BatteryState(20.0)
    _, results = one_house_hour("over_discharge_rate", proposed=-100.0, base=12.0, battery=battery)
    assert results[0].market_load == 2.0
    assert battery.energy == 10.0
    assert results[0].warning == "listed consumption exceeds max battery discharge rate"

    _, results = one_house_hour(
        "exact_charge_boundary",
        proposed=7.0,
        base=2.0,
        battery=BatteryState(0.0),
    )
    assert results[0].market_load == 7.0
    assert results[0].warning == ""
    assert_value_error(lambda: step_market_hour(hour=0, price=math.nan, house_inputs=[]), "price")
    assert_value_error(
        lambda: one_house_hour(
            "test",
            proposed=math.nan,
            base=2.0,
            battery=BatteryState(0.0),
        ),
        "market_load",
    )
    assert_value_error(
        lambda: run_scenario(MarketScenario(policies=[], demand_profile=[1.0])),
        "demand_profile",
    )


def one_house_hour(
    name: str,
    proposed: float,
    base: float,
    battery: BatteryState,
):
    return step_market_hour(
        hour=0,
        price=0.5,
        house_inputs=[
            HouseHourInput(
                name=name,
                proposed_market_load=proposed,
                base_demand=base,
                battery=battery,
            )
        ],
    )


def assert_value_error(call, expected: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError(f"expected ValueError containing {expected!r}")


def run_import_safety_check() -> None:
    helics_path = Path(__file__).resolve().parents[1] / "helics"
    retired_runtime_files = ("market_maker.py", "house_template.py", "battery.py")
    for filename in retired_runtime_files:
        if (helics_path / filename).exists():
            raise AssertionError(
                f"retired downstream HELICS runtime file still exists: {filename}"
            )

    for module_name in (
        "python.market_game_downstream.core.rules",
        "python.market_game_downstream.core.simulator",
        "python.market_game_downstream.rl.evaluate_submission",
    ):
        importlib.import_module(module_name)


if __name__ == "__main__":
    run_shared_core_check()
    run_import_safety_check()
    print("shared core: ok")
