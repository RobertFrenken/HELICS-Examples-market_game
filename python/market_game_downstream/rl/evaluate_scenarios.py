"""CSV evaluator for downstream competition scenarios."""

from __future__ import annotations

import argparse
import csv
import sys

from .scenarios import (
    evaluate_curriculum,
    evaluate_scenario,
    stock_example_scenario,
    weekly_training_scenarios,
)


CSV_COLUMNS = [
    "scenario",
    "agent",
    "profile_type",
    "seed",
    "total_load",
    "total_cost",
    "final_battery",
    "boundary_warnings",
    "clamps",
]


def write_rows(rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def rows_for_args(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.stock:
        return evaluate_scenario(stock_example_scenario())
    if args.scenario:
        scenarios = weekly_training_scenarios(seed=args.seed)
        scenarios_by_name = {scenario.name: scenario for scenario in scenarios}
        if args.scenario not in scenarios_by_name:
            names = ", ".join(scenarios_by_name)
            raise SystemExit(f"unknown scenario {args.scenario!r}; choices: {names}")
        return evaluate_scenario(scenarios_by_name[args.scenario])
    return evaluate_curriculum(seed=args.seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="evaluate downstream market-game competition scenarios as CSV"
    )
    parser.add_argument(
        "--stock",
        action="store_true",
        help="evaluate only the stock profile1 example scenario",
    )
    parser.add_argument(
        "--scenario",
        help="evaluate one named weekly scenario instead of the full curriculum",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed for generated weekly demand profiles and stochastic opponents",
    )
    args = parser.parse_args()
    write_rows(rows_for_args(args))


if __name__ == "__main__":
    main()
