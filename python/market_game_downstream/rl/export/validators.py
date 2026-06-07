"""Static and simulator-based checks for exported market-game submissions."""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import importlib.util
import inspect
import math
from pathlib import Path

from python.market_game_downstream.core.simulator import (
    MarketScenario,
    run_scenario,
)
from python.market_game_downstream.rl.agents.policies import (
    FollowDemandPolicy,
    PriceAwarePolicy,
)
from .export_policy import ComputeDemand, FunctionSubmissionPolicy


EXPECTED_PARAMETERS = (
    "price",
    "hour",
    "battery_charge",
    "demand",
    "price_history",
)

MAX_SOURCE_BYTES = 64_000
ALLOWED_IMPORT_MODULES = frozenset({"math"})
ALLOWED_TOP_LEVEL_NODES = (
    ast.Expr,
    ast.FunctionDef,
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
)
DISALLOWED_CALL_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)
DEFAULT_OPPONENT_FACTORIES: tuple[Callable[[], object], ...] = (
    lambda: FollowDemandPolicy(name="FollowDemandOpponent"),
    lambda: PriceAwarePolicy(name="PriceAwareOpponent"),
)


class SubmissionValidationError(ValueError):
    """Raised when a submission cannot be treated as competition-safe."""


@dataclass(frozen=True)
class ValidationReport:
    """Summary from static checks and a pure-simulator smoke run."""

    name: str
    hours: int
    total_cost: float
    total_load: float
    final_battery: float
    clamps: int
    boundary_warnings: list[str] = field(default_factory=list)
    scenario_count: int = 1

    @property
    def ok(self) -> bool:
        return self.hours == 24 and self.clamps == 0


def load_compute_demand(path: str | Path) -> ComputeDemand:
    """Load ``compute_demand`` from a Python source file."""

    module_path = Path(path)
    if not module_path.is_file():
        raise SubmissionValidationError(f"submission file does not exist: {module_path}")
    if module_path.stat().st_size > MAX_SOURCE_BYTES:
        raise SubmissionValidationError(
            f"submission source exceeds {MAX_SOURCE_BYTES} bytes"
        )
    _validate_source_ast(module_path.read_text(encoding="utf-8"), module_path)
    module = _load_source_module(module_path)
    compute_demand = getattr(module, "compute_demand", None)
    if not callable(compute_demand):
        raise SubmissionValidationError("submission must define callable compute_demand")
    return compute_demand


def validate_submission_file(path: str | Path) -> ValidationReport:
    """Load and validate a source-file submission."""

    compute_demand = load_compute_demand(path)
    return validate_compute_demand(compute_demand)


def validate_compute_demand(
    compute_demand: ComputeDemand,
    opponent_factories: Sequence[Callable[[], object]] | None = None,
) -> ValidationReport:
    """Check signature, numeric output, and 24-hour simulator behavior."""

    _validate_signature(compute_demand)
    _validate_sample_outputs(compute_demand)

    opponents = opponent_factories or DEFAULT_OPPONENT_FACTORIES
    policies = [FunctionSubmissionPolicy(compute_demand), *[factory() for factory in opponents]]
    result = run_scenario(MarketScenario(policies=policies))
    submitted = result.houses[0]
    if len(result.records) != 24:
        raise SubmissionValidationError("submission did not complete a 24-hour simulation")
    return ValidationReport(
        name=submitted.policy.name,
        hours=len(result.records),
        total_cost=submitted.total_cost,
        total_load=submitted.total_load,
        final_battery=submitted.battery.energy,
        clamps=submitted.clamps,
        boundary_warnings=list(submitted.boundary_warnings),
    )


def _validate_signature(compute_demand: ComputeDemand) -> None:
    signature = inspect.signature(compute_demand)
    parameters = tuple(signature.parameters.values())
    if tuple(parameter.name for parameter in parameters) != EXPECTED_PARAMETERS:
        raise SubmissionValidationError(
            "compute_demand signature must be "
            "compute_demand(price, hour, battery_charge, demand, price_history)"
        )
    for parameter in parameters:
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            raise SubmissionValidationError("compute_demand must use five positional parameters")


def _validate_sample_outputs(compute_demand: ComputeDemand) -> None:
    demand = [2.0] * 24
    price_history = [0.5]
    for hour in (0, 7, 17, 23):
        _validate_numeric_output(
            compute_demand(0.25, hour, 5.0, list(demand), list(price_history))
        )


def _validate_source_ast(source: str, module_path: Path) -> None:
    tree = ast.parse(source, filename=str(module_path))
    if not any(_is_function_named(node, "compute_demand") for node in tree.body):
        raise SubmissionValidationError("submission must define compute_demand")

    for node in tree.body:
        _validate_top_level_node(node)

    for node in ast.walk(tree):
        _validate_ast_node(node)


def _load_source_module(module_path: Path) -> object:
    spec = importlib.util.spec_from_file_location("_market_game_submission", module_path)
    if spec is None or spec.loader is None:
        raise SubmissionValidationError(f"cannot load submission file: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_numeric_output(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubmissionValidationError("compute_demand must return a numeric market demand")
    if not math.isfinite(float(value)):
        raise SubmissionValidationError("compute_demand must return a finite number")


def _is_function_named(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.FunctionDef) and node.name == name


def _validate_top_level_node(node: ast.AST) -> None:
    if not isinstance(node, ALLOWED_TOP_LEVEL_NODES):
        raise SubmissionValidationError(
            "submission top level may only contain imports, constants, "
            "and function definitions"
        )
    if isinstance(node, ast.FunctionDef):
        _validate_function_def_import_safety(node)
    if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
        raise SubmissionValidationError(
            "submission must not execute runtime expressions at import time"
        )
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and not _is_literal_assignment(node):
        raise SubmissionValidationError(
            "submission top-level assignments must be literal constants"
        )


def _validate_function_def_import_safety(node: ast.FunctionDef) -> None:
    """Reject function syntax that evaluates user code while importing."""
    if node.decorator_list:
        raise SubmissionValidationError("submission functions must not use decorators")
    if node.args.defaults or any(default is not None for default in node.args.kw_defaults):
        raise SubmissionValidationError("submission functions must not use default arguments")
    if node.returns is not None:
        raise SubmissionValidationError("submission functions must not use annotations")
    all_args = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    if node.args.vararg is not None:
        all_args.append(node.args.vararg)
    if node.args.kwarg is not None:
        all_args.append(node.args.kwarg)
    if any(arg.annotation is not None for arg in all_args):
        raise SubmissionValidationError("submission functions must not use annotations")


def _validate_ast_node(node: ast.AST) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            _validate_import_name(alias.name)
    elif isinstance(node, ast.ImportFrom):
        _validate_import_name(node.module or "")
    elif isinstance(node, ast.Call):
        call_name = _call_name(node.func)
        if call_name in DISALLOWED_CALL_NAMES:
            raise SubmissionValidationError(
                f"disallowed runtime call in submission: {call_name}"
            )
    elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
        raise SubmissionValidationError(
            f"disallowed dunder attribute access in submission: {node.attr}"
        )


def _validate_import_name(module_name: str) -> None:
    if module_name.partition(".")[0] not in ALLOWED_IMPORT_MODULES:
        raise SubmissionValidationError(
            f"disallowed import in submission: {module_name}"
        )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_literal_assignment(node: ast.Assign | ast.AnnAssign) -> bool:
    value = node.value
    if value is None:
        return True
    try:
        ast.literal_eval(value)
    except (ValueError, TypeError):
        return False
    return True
