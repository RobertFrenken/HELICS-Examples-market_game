"""Evaluation helpers for the pure market-game simulator."""

from __future__ import annotations

from .policies import FlattenDemandPolicy, FullCyclePolicy, PriceAwarePolicy
from .simulator import run_episode


def run_stock_example() -> None:
    result = run_episode(
        [
            FlattenDemandPolicy(),
            FullCyclePolicy(),
            PriceAwarePolicy(),
        ]
    )

    print("agent,total_load,total_cost,final_battery")
    for house in result.houses:
        print(
            f"{house.policy.name},"
            f"{house.total_load:.10f},"
            f"{house.total_cost:.10f},"
            f"{house.battery.energy:.10f}"
        )


if __name__ == "__main__":
    run_stock_example()
