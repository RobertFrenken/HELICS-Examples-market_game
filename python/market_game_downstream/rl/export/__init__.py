"""Export and validation helpers for competition-safe submissions."""

from .distill import (
    DecisionTreePolicy,
    DistillationReport,
    MatrixPolicy,
    TeacherSample,
    ThresholdRule,
    collect_teacher_samples,
    fit_decision_tree_policy,
    fit_matrix_policy,
    fit_threshold_rule,
    render_matrix_submission,
    render_tree_submission,
    render_threshold_submission,
    write_matrix_submission,
    write_threshold_submission,
    write_tree_submission,
)
from .export_policy import FunctionSubmissionPolicy, policy_to_compute_demand
from .validators import (
    SubmissionValidationError,
    ValidationReport,
    load_compute_demand,
    validate_compute_demand,
    validate_submission_file,
)

__all__ = [
    "DistillationReport",
    "FunctionSubmissionPolicy",
    "DecisionTreePolicy",
    "MatrixPolicy",
    "SubmissionValidationError",
    "TeacherSample",
    "ThresholdRule",
    "ValidationReport",
    "collect_teacher_samples",
    "fit_decision_tree_policy",
    "fit_matrix_policy",
    "fit_threshold_rule",
    "load_compute_demand",
    "policy_to_compute_demand",
    "render_matrix_submission",
    "render_tree_submission",
    "render_threshold_submission",
    "validate_compute_demand",
    "validate_submission_file",
    "write_matrix_submission",
    "write_threshold_submission",
    "write_tree_submission",
]
