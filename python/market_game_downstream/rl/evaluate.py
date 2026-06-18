"""Single CSV evaluator for RL scenarios and standalone submissions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from python.market_game_downstream.core import (
    HourRecord,
    HousePolicy,
    SimulationResult,
    run_scenario,
)

from .export.export_policy import FunctionSubmissionPolicy
from .export.validators import SubmissionValidationError, validate_submission_file
from .envs.scenarios import (
    CompetitionScenario,
    PolicyFactory,
    format_scenario_choices,
    scenario_by_name,
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
    "invalid_load_adjustment",
    "penalty_cost",
    "price_volatility",
]


def rows_for_args(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.stock:
        return _evaluate_selected(args, [stock_example_scenario()])
    return [
        row
        for seed in evaluation_seeds(args)
        for row in _rows_for_seed(args, seed)
    ]


def evaluation_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return _parse_seed_list(args.seeds)
    return [args.seed]


def write_rows(rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def evaluate_scenario(scenario: CompetitionScenario) -> list[dict[str, str]]:
    result = run_scenario(scenario.to_market_scenario())
    volatility = result_price_volatility(result)
    return [
        _summary_row(scenario, house, result.records, volatility)
        for house in result.houses
    ]


def evaluate_curriculum(seed: int = 1) -> list[dict[str, str]]:
    return [
        row
        for scenario in weekly_training_scenarios(seed=seed)
        for row in evaluate_scenario(scenario)
    ]


def submitted_function_policy_factory(
    path: str | Path,
    name: str = "SubmittedHouse",
) -> PolicyFactory:
    if not isinstance(name, str) or not name:
        raise ValueError("submitted function policy name must be a non-empty string")
    submission_path = Path(path)

    def factory(path: Path = submission_path, name: str = name) -> HousePolicy:
        from .export.validators import load_compute_demand

        return FunctionSubmissionPolicy(load_compute_demand(path), name=name)

    return factory


def price_volatility(prices: list[float]) -> float:
    """Sum absolute hour-to-hour price movements."""
    if len(prices) < 2:
        return 0.0
    return sum(abs(prices[index] - prices[index - 1]) for index in range(1, len(prices)))


def result_price_volatility(result: SimulationResult) -> float:
    return price_volatility(result.price_history)


def _summary_row(
    scenario: CompetitionScenario,
    house: object,
    records: list[HourRecord],
    volatility: float,
) -> dict[str, str]:
    penalty_cost = sum(
        record.penalties_by_house.get(house.policy.name, 0.0)
        for record in records
    )
    invalid_load_adjustment = sum(
        record.invalid_load_adjustments_by_house.get(house.policy.name, 0.0)
        for record in records
    )
    return {
        "scenario": scenario.name,
        "agent": house.policy.name,
        "profile_type": scenario.profile_type,
        "seed": str(scenario.seed),
        "total_load": f"{house.total_load:.10f}",
        "total_cost": f"{house.total_cost:.10f}",
        "final_battery": f"{house.battery.energy:.10f}",
        "boundary_warnings": str(len(house.boundary_warnings)),
        "clamps": str(house.clamps),
        "invalid_load_adjustment": f"{invalid_load_adjustment:.10f}",
        "penalty_cost": f"{penalty_cost:.10f}",
        "price_volatility": f"{volatility:.10f}",
    }


def _rows_for_seed(args: argparse.Namespace, seed: int) -> list[dict[str, str]]:
    scenarios = weekly_training_scenarios(seed=seed)
    if args.scenario:
        try:
            return _evaluate_selected(args, [scenario_by_name(args.scenario, scenarios)])
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    return _evaluate_selected(args, scenarios)


def _evaluate_selected(
    args: argparse.Namespace,
    scenarios: list[CompetitionScenario],
) -> list[dict[str, str]]:
    rows = [
        row
        for scenario in scenarios
        for row in evaluate_scenario(_scenario_for_args(args, scenario))
    ]
    if args.only_submission:
        rows = [row for row in rows if row["agent"] == args.submission_name]
    return rows


def _scenario_for_args(
    args: argparse.Namespace,
    scenario: CompetitionScenario,
) -> CompetitionScenario:
    if not args.submission:
        return scenario
    return scenario.with_policy_factory(
        submitted_function_policy_factory(args.submission, name=args.submission_name),
        first=True,
    )


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
    parser = argparse.ArgumentParser(description="evaluate market-game RL scenarios")
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="print built-in scenario IDs and exit",
    )
    parser.add_argument("--stock", action="store_true", help="evaluate the stock parity scenario")
    parser.add_argument("--scenario", help="evaluate one named scenario")
    parser.add_argument("--seed", type=int, default=1, help="scenario seed")
    parser.add_argument("--seeds", help="comma-separated seed list, e.g. 1,2,3")
    parser.add_argument(
        "--submission",
        help="path to a standalone file defining compute_demand(...)",
    )
    parser.add_argument(
        "--submission-name",
        default="SubmittedHouse",
        help="agent name used for --submission rows",
    )
    parser.add_argument(
        "--only-submission",
        action="store_true",
        help="with --submission, emit only the submitted policy row",
    )
    parser.add_argument(
        "--validate-submission",
        action="store_true",
        help="validate --submission before scoring",
    )
    args = parser.parse_args()
    if args.list_scenarios:
        print(format_scenario_choices(seed=args.seed))
        return
    if args.validate_submission:
        if not args.submission:
            raise SystemExit("--validate-submission requires --submission")
        report = validate_submission_file(args.submission)
        if not report.ok:
            raise SystemExit(
                "submission failed validation "
                f"(hours={report.hours}, clamps={report.clamps})"
            )
    try:
        write_rows(rows_for_args(args))
    except SubmissionValidationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
