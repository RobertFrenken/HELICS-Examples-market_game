"""Smoke check for competition submission export helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile

from python.market_game_downstream.rl.agents.policies import PriceAwarePolicy
from python.market_game_downstream.rl.export import (
    SubmissionValidationError,
    ValidationReport,
    collect_teacher_samples,
    fit_decision_tree_policy,
    fit_matrix_policy,
    fit_threshold_rule,
    policy_to_compute_demand,
    validate_compute_demand,
    validate_submission_file,
    write_matrix_submission,
    write_threshold_submission,
    write_tree_submission,
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

    def teacher_action(obs: object) -> int:
        price = float(obs[2])
        battery_charge = float(obs[3])
        if price <= 0.12:
            return 2
        if price >= 0.49 and battery_charge > 0.0:
            return 0
        return 1

    samples = collect_teacher_samples(
        teacher_action,
        scenario_names=["week_1_baselines"],
        observation_mode="local",
        scenario_seed=3,
    )
    distillation = fit_threshold_rule(samples)
    assert distillation.samples == 24
    assert distillation.accuracy >= 0.8
    tree_distillation = fit_decision_tree_policy(
        samples,
        observation_mode="local",
        max_depth=3,
    )
    assert tree_distillation.samples == 24
    assert tree_distillation.accuracy >= 0.8
    matrix_distillation = fit_matrix_policy(
        samples,
        observation_mode="local",
        epochs=100,
    )
    assert matrix_distillation.samples == 24
    assert matrix_distillation.accuracy >= 0.7

    with tempfile.TemporaryDirectory() as temp_dir:
        distilled_submission = Path(temp_dir) / "distilled_submission.py"
        write_threshold_submission(distillation.rule, distilled_submission)
        assert_valid_report(validate_submission_file(distilled_submission))
        tree_submission = Path(temp_dir) / "tree_submission.py"
        write_tree_submission(tree_distillation.rule, tree_submission)
        assert_valid_report(validate_submission_file(tree_submission))
        matrix_submission = Path(temp_dir) / "matrix_submission.py"
        write_matrix_submission(matrix_distillation.rule, matrix_submission)
        assert_valid_report(validate_submission_file(matrix_submission))

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
