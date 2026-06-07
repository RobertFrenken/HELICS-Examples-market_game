"""CSV evaluator for downstream competition scenarios."""

from __future__ import annotations

import argparse
import csv
import sys

from .scenarios import (
    evaluate_scenario,
    load_scenarios,
    scenario_by_name,
    stock_example_scenario,
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
    scenarios = load_scenarios(args.config, seed=args.seed)
    if args.scenario:
        try:
            scenario = scenario_by_name(args.scenario, scenarios)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return evaluate_scenario(scenario)
    return [
        row
        for scenario in scenarios
        for row in evaluate_scenario(scenario)
    ]


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
    parser.add_argument(
        "--config",
        default=None,
        help="JSON scenario config path; defaults to rl/scenario_configs/weekly.json",
    )
    args = parser.parse_args()
    write_rows(rows_for_args(args))


if __name__ == "__main__":
    main()
