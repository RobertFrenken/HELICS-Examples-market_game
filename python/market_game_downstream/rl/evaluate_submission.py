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
    submission_name = args.name
    scenario_args = argparse.Namespace(
        stock=False,
        scenario=args.scenario,
        seed=args.seed,
        seeds=None,
        config=args.config,
        submission=args.submission,
        submission_name=submission_name,
        only_submission=not getattr(args, "include_baselines", False),
    )
    rows = rows_for_args(scenario_args)
    submitted_rows = [row for row in rows if row["agent"] == submission_name]
    if len(submitted_rows) != 1:
        raise SubmissionValidationError(
            f"expected exactly one scored row for submission {submission_name!r}"
        )
    _validate_scored_submission_row(submitted_rows[0])
    return [_practice_row(row) for row in rows]


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


def _validate_scored_submission_row(row: dict[str, str]) -> None:
    clamps = _int_field(row, "clamps")
    if clamps:
        raise SubmissionValidationError(
            "submission failed scored scenario validation "
            f"(scenario={row['scenario']}, clamps={clamps})"
        )


def _int_field(row: dict[str, str], name: str) -> int:
    try:
        return int(row[name])
    except (KeyError, ValueError) as exc:
        raise SubmissionValidationError(
            f"submission scenario row has invalid {name!r}: {row.get(name)!r}"
        ) from exc


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
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="include compact rows for the scenario baseline opponents",
    )
    args = parser.parse_args()
    try:
        write_practice_rows(practice_rows(args))
    except SubmissionValidationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
