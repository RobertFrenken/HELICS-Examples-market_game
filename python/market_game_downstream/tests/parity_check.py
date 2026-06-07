"""Executable parity assertions for the stock market-game example."""

from __future__ import annotations

from python.market_game_downstream.rl.core.simulator import run_episode
from python.market_game_downstream.rl.evaluate import stock_scenario


EXPECTED_STOCK = {
    "FlattenDemandHouse": (127.0, 35.446666666666665, 7.0),
    "FullCycleHouse": (120.0, 47.74666666666667, 0.0),
    "PriceAwareHouse": (125.0, 21.953333333333333, 5.0),
}


def assert_stock_parity() -> None:
    policies = [factory() for factory in stock_scenario().policy_factories]
    result = run_episode(policies)
    for house in result.houses:
        expected_load, expected_cost, expected_battery = EXPECTED_STOCK[house.policy.name]
        assert abs(house.total_load - expected_load) < 1e-9, house.policy.name
        assert abs(house.total_cost - expected_cost) < 1e-9, house.policy.name
        assert abs(house.battery.energy - expected_battery) < 1e-9, house.policy.name


if __name__ == "__main__":
    assert_stock_parity()
    print("stock parity: ok")
