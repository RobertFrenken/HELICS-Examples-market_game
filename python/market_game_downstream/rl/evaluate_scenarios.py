"""CSV evaluator for downstream competition scenarios."""

from __future__ import annotations

import argparse
import csv
import sys

from .scenarios import (
    CompetitionScenario,
    evaluate_scenario,
    load_scenarios,
    scenario_by_name,
    submitted_function_policy_factory,
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
        return _evaluate_selected_scenarios(args, [stock_example_scenario()])
    return [
        row
        for seed in evaluation_seeds(args)
        for row in _rows_for_seed(args, seed)
    ]


def _rows_for_seed(args: argparse.Namespace, seed: int) -> list[dict[str, str]]:
    scenarios = load_scenarios(args.config, seed=seed)
    if args.scenario:
        try:
            scenario = scenario_by_name(args.scenario, scenarios)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return _evaluate_selected_scenarios(args, [scenario])
    return _evaluate_selected_scenarios(args, scenarios)


def _evaluate_selected_scenarios(
    args: argparse.Namespace,
    scenarios: list[CompetitionScenario],
) -> list[dict[str, str]]:
    rows = [
        row
        for scenario in scenarios
        for row in evaluate_scenario(_scenario_for_args(args, scenario))
    ]
    if _arg(args, "only_submission", False):
        submission_name = _arg(args, "submission_name", "SubmittedHouse")
        rows = [row for row in rows if row["agent"] == submission_name]
    return rows


def _scenario_for_args(
    args: argparse.Namespace,
    scenario: CompetitionScenario,
) -> CompetitionScenario:
    submission = _arg(args, "submission", None)
    if not submission:
        return scenario
    name = _arg(args, "submission_name", "SubmittedHouse")
    return scenario.with_policy_factory(
        submitted_function_policy_factory(submission, name=name),
        first=True,
    )


def evaluation_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return _parse_seed_list(args.seeds)
    return [args.seed]


def _arg(args: argparse.Namespace, name: str, default: object) -> object:
    return getattr(args, name, default)


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
    parser.add_argument(
        "--submission",
        help="optional path to a standalone file defining compute_demand(...)",
    )
    parser.add_argument(
        "--submission-name",
        default="SubmittedHouse",
        help="agent name used for --submission output rows",
    )
    parser.add_argument(
        "--only-submission",
        action="store_true",
        help="with --submission, emit only the submitted policy row",
    )
    args = parser.parse_args()
    write_rows(rows_for_args(args))


if __name__ == "__main__":
    main()
