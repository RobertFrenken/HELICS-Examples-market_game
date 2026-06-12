"""Checks for canonical house-template helper APIs."""

from __future__ import annotations

from pathlib import Path
import sys


MARKET_GAME_DIR = Path(__file__).resolve().parents[1]
if str(MARKET_GAME_DIR) not in sys.path:
    sys.path.insert(0, str(MARKET_GAME_DIR))

from house_template import (
    ActionHouse,
    BatteryAction,
    DeltaHouse,
    action_to_market_load,
    delta_to_market_load,
)


class ChargeHouse(ActionHouse):
    def choose_action(self, price, hour, battery_charge, demand, price_history):
        return BatteryAction.CHARGE


class BigDeltaHouse(DeltaHouse):
    def choose_delta(self, price, hour, battery_charge, demand, price_history):
        return 100.0


def run_check() -> None:
    assert action_to_market_load(BatteryAction.CHARGE, 4.0, 0.0) == 9.0
    assert action_to_market_load(BatteryAction.DISCHARGE, 12.0, 20.0) == 2.0
    assert action_to_market_load(BatteryAction.NEUTRAL, 6.0, 10.0) == 6.0
    assert action_to_market_load(1, 4.0, 18.0) == 6.0

    assert delta_to_market_load(100.0, 4.0, 18.0) == 6.0
    assert delta_to_market_load(-100.0, 12.0, 4.0) == 8.0
    assert delta_to_market_load(0.0, 6.0, 10.0) == 6.0

    charge_house = object.__new__(ChargeHouse)
    assert charge_house.compute_demand(0.1, 0, 0.0, [4.0], [0.1]) == 9.0

    delta_house = object.__new__(BigDeltaHouse)
    assert delta_house.compute_demand(0.1, 0, 18.0, [4.0], [0.1]) == 6.0


if __name__ == "__main__":
    run_check()
    print("house template: ok")
