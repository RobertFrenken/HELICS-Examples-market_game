"""Checks for the market-game market maker helpers."""

from __future__ import annotations

from pathlib import Path
import math
import random
import sys


MARKET_GAME_DIR = Path(__file__).resolve().parents[1]
if str(MARKET_GAME_DIR) not in sys.path:
    sys.path.insert(0, str(MARKET_GAME_DIR))

from battery import Battery
from market_maker import SubFed, compute_new_price, run_market_hour, update_demand


class FakeHelics:
    """Minimal HELICS shim for testing pure market-hour accounting."""

    @staticmethod
    def helicsInputGetDouble(value: float) -> float:
        return value


def check_price_curve() -> None:
    assert compute_new_price(total=0.0, feds=0) == 0.1
    assert compute_new_price(total=2.0, feds=1) == 0.1
    assert math.isclose(compute_new_price(total=4.0, feds=1), 0.13)
    assert math.isclose(compute_new_price(total=7.0, feds=1), 0.29)
    assert math.isclose(compute_new_price(total=10.0, feds=1), 0.74)
    assert math.isclose(compute_new_price(total=14.0, feds=1), 2.49)


def check_demand_profiles() -> None:
    fed = SubFed()
    update_demand("flat", fed)
    assert fed.demand == [5] * 24

    update_demand("profile1", fed)
    assert len(fed.demand) == 24
    assert fed.demand[0] == 2
    assert fed.demand[17] == 12

    update_demand("profile_solar", fed)
    assert len(fed.demand) == 24
    assert fed.demand[8] < 0

    random.seed(1)
    update_demand("random", fed)
    assert len(fed.demand) == 24
    assert round(sum(fed.demand), 10) == 120.0


def check_market_hour_accounting() -> None:
    high_load = SubFed(input=100.0, name="high_load")
    steady_load = SubFed(input=5.0, battery=Battery(5.0), name="steady_load")
    feds = [high_load, steady_load]

    total_load = run_market_hour(FakeHelics, feds, hour=0, current_price=0.5)

    assert total_load == 15.0

    assert high_load.consume == [10.0]
    assert high_load.battery.energy == 5.0
    assert high_load.hourCost == [1805.0]

    assert steady_load.consume == [5.0]
    assert steady_load.battery.energy == 5.0
    assert steady_load.hourCost == [2.5]


def run_check() -> None:
    check_price_curve()
    check_demand_profiles()
    check_market_hour_accounting()


if __name__ == "__main__":
    run_check()
    print("market maker: ok")
