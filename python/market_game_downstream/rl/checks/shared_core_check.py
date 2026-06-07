"""Checks for shared market-game core compatibility."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

from python.market_game_downstream.core import BatteryAction, action_to_market_load
from python.market_game_downstream.core.config import demand_profile
from python.market_game_downstream.core.rules import check_valid, compute_price_from_total_load, ensure_valid
from python.market_game_downstream.core.simulator import BatteryState, HouseHourInput, MarketScenario, step_market_hour


def run_shared_core_check() -> None:
    assert compute_price_from_total_load(15.0, 3) == 0.16
    assert demand_profile("profile1")[0:4] == [2, 1, 1, 1]

    battery = BatteryState(0.0)
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


def run_import_safety_check() -> None:
    helics_path = Path(__file__).resolve().parents[2] / "helics"
    if str(helics_path) not in sys.path:
        sys.path.insert(0, str(helics_path))

    for module_name in ("market_maker", "house_template", "battery"):
        importlib.import_module(module_name)


if __name__ == "__main__":
    run_shared_core_check()
    run_import_safety_check()
    print("shared core: ok")
