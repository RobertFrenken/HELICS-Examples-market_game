"""Smoke check for competition submission export helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile

from python.market_game_downstream.rl.agents.policies import PriceAwarePolicy
from python.market_game_downstream.rl.export import (
    SubmissionValidationError,
    ValidationReport,
    policy_to_compute_demand,
    validate_compute_demand,
    validate_submission_file,
)


def assert_valid_report(report: ValidationReport) -> None:
    assert report.hours == 24
    assert report.clamps == 0
    assert report.ok


def run_export_smoke_check() -> None:
    export_dir = Path(__file__).resolve().parents[1] / "rl" / "export"
    assert_valid_report(
        validate_submission_file(export_dir / "example_threshold_submission.py")
    )

    compute_demand = policy_to_compute_demand(PriceAwarePolicy())
    assert_valid_report(validate_compute_demand(compute_demand))

    with tempfile.TemporaryDirectory() as temp_dir:
        bad_submission = Path(temp_dir) / "bad_submission.py"
        bad_submission.write_text(
            "print('import-time side effect')\n"
            "def compute_demand(price, hour, battery_charge, demand, price_history):\n"
            "    return globals()['x']\n",
            encoding="utf-8",
        )
        try:
            validate_submission_file(bad_submission)
        except SubmissionValidationError:
            pass
        else:
            raise AssertionError("unsafe submission was not rejected")


if __name__ == "__main__":
    run_export_smoke_check()
    print("export smoke: ok")
