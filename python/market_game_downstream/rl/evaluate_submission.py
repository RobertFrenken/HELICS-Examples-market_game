"""Compact evaluator for standalone market-game submission files."""

from __future__ import annotations

import argparse
import csv
import sys

from .export.validators import SubmissionValidationError, validate_submission_file
from .evaluate_scenarios import rows_for_args


PRACTICE_COLUMNS = [
    "house",
    "total_load",
    "total_cost",
    "final_battery",
    "clamps",
]


def practice_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    report = validate_submission_file(args.submission)
    if not report.ok:
        raise SubmissionValidationError(
            "submission failed 24-hour validation "
            f"(hours={report.hours}, clamps={report.clamps})"
        )
    scenario_args = argparse.Namespace(
        stock=False,
        scenario=args.scenario,
        seed=args.seed,
        seeds=None,
        config=args.config,
        submission=args.submission,
        submission_name=args.name,
        only_submission=True,
    )
    return [_practice_row(row) for row in rows_for_args(scenario_args)]


def write_practice_rows(rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=PRACTICE_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _practice_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "house": row["agent"],
        "total_load": row["total_load"],
        "total_cost": row["total_cost"],
        "final_battery": row["final_battery"],
        "clamps": row["clamps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="evaluate one standalone compute_demand(...) submission"
    )
    parser.add_argument(
        "submission",
        help="path to a standalone file defining compute_demand(...)",
    )
    parser.add_argument(
        "--name",
        default="SubmittedHouse",
        help="house name used in output",
    )
    parser.add_argument(
        "--scenario",
        default="week_1_baselines",
        help="scenario to evaluate; defaults to week_1_baselines",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed for generated demand profiles and stochastic opponents",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON scenario config path; defaults to rl/scenario_configs/weekly.json",
    )
    args = parser.parse_args()
    try:
        write_practice_rows(practice_rows(args))
    except SubmissionValidationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
