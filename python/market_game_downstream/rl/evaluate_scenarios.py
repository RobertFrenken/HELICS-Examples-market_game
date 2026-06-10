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
    "invalid_load_adjustment",
    "penalty_cost",
    "price_volatility",
]


def write_rows(rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def rows_for_args(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.stock:
        return evaluate_scenario(stock_example_scenario())
    return [
        row
        for seed in _evaluation_seeds(args)
        for row in _rows_for_seed(args, seed)
    ]


def _rows_for_seed(args: argparse.Namespace, seed: int) -> list[dict[str, str]]:
    scenarios = load_scenarios(args.config, seed=seed)
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


def _evaluation_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return _parse_seed_list(args.seeds)
    return [args.seed]


def _parse_seed_list(value: str) -> list[int]:
    seeds: list[int] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            raise SystemExit("--seeds must be a comma-separated list of integers")
        try:
            seeds.append(int(text))
        except ValueError as exc:
            raise SystemExit(f"invalid seed in --seeds: {text!r}") from exc
    return seeds


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
        "--seeds",
        help="comma-separated seed list for repeated scenario evaluation, e.g. 1,2,3",
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
