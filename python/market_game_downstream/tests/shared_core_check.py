"""Checks for shared market-game core compatibility."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys

from python.market_game_downstream.core import BatteryAction, action_to_market_load
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
    try:
        demand_profile("typo")
    except ValueError as exc:
        assert "unknown demand profile" in str(exc)
    else:
        raise AssertionError("unknown demand profile was not rejected")

    try:
        MarketGameConfig(episode_hours=25, demand_profile=[1.0] * 24)
    except ValueError as exc:
        assert "demand_profile" in str(exc)
    else:
        raise AssertionError("short demand profile was not rejected")
    try:
        compute_price_from_total_load(math.nan, 1)
    except ValueError as exc:
        assert "total_market_load" in str(exc)
    else:
        raise AssertionError("non-finite total load was not rejected")

    battery = BatteryState(0.0)
    try:
        battery.change(math.nan)
    except ValueError as exc:
        assert "battery delta" in str(exc)
    else:
        raise AssertionError("non-finite battery delta was not rejected")
    warning = check_valid(-1.0, 2.0, battery)
    assert warning == "listed consumption exceeds available battery energy"
    assert ensure_valid(-1.0, 2.0, battery) == 2.0

    assert action_to_market_load(BatteryAction.CHARGE, 2.0, 0.0) == 7.0
    assert action_to_market_load(BatteryAction.DISCHARGE, 2.0, 0.0) == 2.0

    scenario = MarketScenario(policies=[])
    assert scenario.config.episode_hours == 24

    battery = BatteryState(0.0)
    record, results = step_market_hour(
        hour=0,
        price=0.5,
        house_inputs=[
            HouseHourInput(
                name="test",
                proposed_market_load=-1.0,
                base_demand=2.0,
                battery=battery,
            )
        ],
    )
    assert results[0].proposed_market_load == -1.0
    assert results[0].market_load == 2.0
    assert record.proposed_loads_by_house["test"] == -1.0
    assert record.loads_by_house["test"] == 2.0
    assert record.warnings_by_house["test"] == "listed consumption exceeds available battery energy"
    try:
        step_market_hour(
            hour=0,
            price=math.nan,
            house_inputs=[],
        )
    except ValueError as exc:
        assert "price" in str(exc)
    else:
        raise AssertionError("non-finite market price was not rejected")
    try:
        step_market_hour(
            hour=0,
            price=0.5,
            house_inputs=[
                HouseHourInput(
                    name="test",
                    proposed_market_load=math.nan,
                    base_demand=2.0,
                    battery=BatteryState(0.0),
                )
            ],
        )
    except ValueError as exc:
        assert "market_load" in str(exc)
    else:
        raise AssertionError("non-finite market load was not rejected")
    try:
        run_scenario(MarketScenario(policies=[], demand_profile=[1.0]))
    except ValueError as exc:
        assert "demand_profile" in str(exc)
    else:
        raise AssertionError("short scenario demand profile was not rejected")


def run_import_safety_check() -> None:
    helics_path = Path(__file__).resolve().parents[1] / "helics"
    if str(helics_path) not in sys.path:
        sys.path.insert(0, str(helics_path))

    for module_name in ("market_maker", "house_template", "battery"):
        importlib.import_module(module_name)


if __name__ == "__main__":
    run_shared_core_check()
    run_import_safety_check()
    print("shared core: ok")
