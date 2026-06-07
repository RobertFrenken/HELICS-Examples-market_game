"""CLI for distilling an RLlib checkpoint into a small submission policy."""

from __future__ import annotations

import argparse
import logging
import os

import ray

from ..agents.observations import ObservationMode
from .distill import (
    DecisionTreePolicy,
    MatrixPolicy,
    ThresholdRule,
    collect_teacher_samples,
    fit_decision_tree_policy,
    fit_matrix_policy,
    fit_threshold_rule,
    load_teacher_from_checkpoint,
    write_matrix_submission,
    write_threshold_submission,
    write_tree_submission,
)
from .validators import validate_submission_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="distill a trained RLlib checkpoint into a compute_demand file"
    )
    parser.add_argument("--checkpoint", required=True, help="RLlib checkpoint path")
    parser.add_argument(
        "--observation-mode",
        choices=[mode.value for mode in ObservationMode],
        default=ObservationMode.PRICE_HISTORY.value,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        help="scenario to collect teacher actions from; may be provided more than once",
    )
    parser.add_argument(
        "--scenario-config",
        help="JSON scenario config path; defaults to rl/scenario_configs/weekly.json",
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        help="seed override for generated scenario demand profiles and stochastic opponents",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="path for the distilled self-contained compute_demand file",
    )
    parser.add_argument(
        "--student",
        choices=["threshold", "tree", "matrix"],
        default="threshold",
        help="student policy representation to distill",
    )
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--matrix-epochs", type=int, default=800)
    args = parser.parse_args()

    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level=logging.ERROR,
        num_cpus=1,
    )
    algorithm = None
    try:
        algorithm, teacher_action = load_teacher_from_checkpoint(
            args.checkpoint,
            observation_mode=args.observation_mode,
        )
        samples = collect_teacher_samples(
            teacher_action,
            scenario_names=args.scenario,
            observation_mode=args.observation_mode,
            scenario_config=args.scenario_config,
            scenario_seed=args.scenario_seed,
        )
        if args.student == "tree":
            report = fit_decision_tree_policy(
                samples,
                observation_mode=args.observation_mode,
                max_depth=args.tree_depth,
            )
            if not isinstance(report.rule, DecisionTreePolicy):
                raise TypeError("tree distillation returned unexpected policy type")
            output_path = write_tree_submission(report.rule, args.output)
        elif args.student == "matrix":
            report = fit_matrix_policy(
                samples,
                observation_mode=args.observation_mode,
                epochs=args.matrix_epochs,
            )
            if not isinstance(report.rule, MatrixPolicy):
                raise TypeError("matrix distillation returned unexpected policy type")
            output_path = write_matrix_submission(report.rule, args.output)
        else:
            report = fit_threshold_rule(samples)
            if not isinstance(report.rule, ThresholdRule):
                raise TypeError("threshold distillation returned unexpected policy type")
            output_path = write_threshold_submission(report.rule, args.output)
        validation = validate_submission_file(output_path)
        details = _student_details(report.rule)
        print(
            "distilled "
            f"student={args.student} "
            f"samples={report.samples} "
            f"matches={report.matches} "
            f"accuracy={report.accuracy:.4f} "
            f"{details} "
            f"output={output_path} "
            f"validation_cost={validation.total_cost:.10f}"
        )
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


def _student_details(rule: object) -> str:
    if isinstance(rule, ThresholdRule):
        return (
            f"cheap_price={rule.cheap_price:.4f} "
            f"expensive_price={rule.expensive_price:.4f} "
            f"late_hour={rule.late_hour}"
        )
    if isinstance(rule, DecisionTreePolicy):
        return f"tree_depth={_tree_depth(rule.root)}"
    if isinstance(rule, MatrixPolicy):
        return f"matrix_shape={len(rule.weights)}x{len(rule.weights[0]) if rule.weights else 0}"
    return "student_details=unknown"


def _tree_depth(node: object) -> int:
    if not hasattr(node, "left") or not hasattr(node, "right"):
        return 0
    left = getattr(node, "left")
    right = getattr(node, "right")
    if left is None or right is None:
        return 1
    return 1 + max(_tree_depth(left), _tree_depth(right))


if __name__ == "__main__":
    main()
