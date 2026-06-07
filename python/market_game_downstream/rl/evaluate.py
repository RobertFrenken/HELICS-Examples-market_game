"""Evaluation helpers for the pure market-game simulator."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from .core.metrics import result_price_volatility
from .agents.policies import (
    FlattenDemandPolicy,
    FollowDemandPolicy,
    FullCyclePolicy,
    LegalInferencePolicy,
    PriceAwarePolicy,
    RollingPricePolicy,
)
from .core.simulator import HousePolicy, run_episode

PolicyFactory = Callable[[], HousePolicy]


@dataclass(frozen=True)
class Scenario:
    name: str
    policy_factories: list[PolicyFactory]


def run_scenario(scenario: Scenario) -> list[dict[str, str]]:
    policies = [factory() for factory in scenario.policy_factories]
    result = run_episode(policies)
    volatility = result_price_volatility(result)
    rows = []
    for house in result.houses:
        rows.append(
            {
                "scenario": scenario.name,
                "agent": house.policy.name,
                "total_load": f"{house.total_load:.10f}",
                "total_cost": f"{house.total_cost:.10f}",
                "final_battery": f"{house.battery.energy:.10f}",
                "boundary_warnings": str(len(house.boundary_warnings)),
                "clamps": str(house.clamps),
                "price_volatility": f"{volatility:.10f}",
            }
        )
    return rows


def stock_scenario() -> Scenario:
    return Scenario(
        name="stock_examples",
        policy_factories=[
            FlattenDemandPolicy,
            FullCyclePolicy,
            PriceAwarePolicy,
        ],
    )


def baseline_scenarios() -> list[Scenario]:
    return [
        stock_scenario(),
        Scenario(
            name="all_follow_demand",
            policy_factories=[
                lambda: FollowDemandPolicy(name="FollowDemandHouse_1"),
                lambda: FollowDemandPolicy(name="FollowDemandHouse_2"),
                lambda: FollowDemandPolicy(name="FollowDemandHouse_3"),
            ],
        ),
        Scenario(
            name="heuristic_mix",
            policy_factories=[
                lambda: FollowDemandPolicy(name="FollowDemandHouse"),
                RollingPricePolicy,
                LegalInferencePolicy,
            ],
        ),
        Scenario(
            name="price_aware_opponents",
            policy_factories=[
                LegalInferencePolicy,
                lambda: PriceAwarePolicy(name="PriceAwareHouse_1"),
                lambda: PriceAwarePolicy(name="PriceAwareHouse_2"),
            ],
        ),
    ]


def print_rows(rows: list[dict[str, str]]) -> None:
    columns = [
        "scenario",
        "agent",
        "total_load",
        "total_cost",
        "final_battery",
        "boundary_warnings",
        "clamps",
        "price_volatility",
    ]
    print(",".join(columns))
    for row in rows:
        print(",".join(row[column] for column in columns))


def run_stock_example() -> None:
    rows = run_scenario(stock_scenario())
    print("agent,total_load,total_cost,final_battery")
    for row in rows:
        print(
            f"{row['agent']},"
            f"{row['total_load']},"
            f"{row['total_cost']},"
            f"{row['final_battery']}"
        )


def run_baselines() -> None:
    rows = []
    for scenario in baseline_scenarios():
        rows.extend(run_scenario(scenario))
    print_rows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate pure market-game policies")
    parser.add_argument("--stock", action="store_true", help="print only the stock HELICS parity scenario")
    args = parser.parse_args()

    if args.stock:
        run_stock_example()
    else:
        run_baselines()


if __name__ == "__main__":
    main()
