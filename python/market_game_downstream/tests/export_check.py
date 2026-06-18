"""Smoke checks for standalone submission validation."""

from __future__ import annotations

from pathlib import Path
import tempfile

from python.market_game_downstream.rl.agents.policies import PriceAwarePolicy
from python.market_game_downstream.rl.export import (
    EXPORTABLE_PPO_PROFILE,
    ExportProfile,
    SubmissionValidationError,
    ValidationReport,
    policy_to_compute_demand,
    validate_compute_demand,
    validate_submission_file,
)
from python.market_game_downstream.rl.export.export_rllib_checkpoint import render_submission_source


def assert_valid_report(report: ValidationReport) -> None:
    assert report.hours == 24
    assert report.clamps == 0
    assert report.ok


def assert_unsafe_submission(source: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "bad_submission.py"
        path.write_text(source, encoding="utf-8")
        try:
            validate_submission_file(path)
        except SubmissionValidationError:
            pass
        else:
            raise AssertionError("unsafe submission was not rejected")


def run_export_smoke_check() -> None:
    export_dir = Path(__file__).resolve().parents[1] / "rl" / "export"
    assert_valid_report(
        validate_submission_file(export_dir / "example_threshold_submission.py")
    )

    compute_demand = policy_to_compute_demand(PriceAwarePolicy())
    assert_valid_report(validate_compute_demand(compute_demand))

    assert_unsafe_submission(
        "print('import-time side effect')\n"
        "def compute_demand(price, hour, battery_charge, demand, price_history):\n"
        "    return globals()['x']\n"
    )
    assert_unsafe_submission(
        "import os\n"
        "def compute_demand(price, hour, battery_charge, demand, price_history):\n"
        "    return demand[hour]\n"
    )
    assert_unsafe_submission(
        "def compute_demand(price, hour, battery_charge, demand, price_history):\n"
        "    return eval('1')\n"
    )

    EXPORTABLE_PPO_PROFILE.validate()
    assert EXPORTABLE_PPO_PROFILE.model_config() == {
        "fcnet_hiddens": [8],
        "fcnet_activation": "tanh",
    }
    tiny_state = {
        "w1": [[0.0] * 10 for _ in range(EXPORTABLE_PPO_PROFILE.hidden_size)],
        "b1": [0.0] * EXPORTABLE_PPO_PROFILE.hidden_size,
        "w2": [[0.0] * EXPORTABLE_PPO_PROFILE.hidden_size for _ in range(3)],
        "b2": [0.0, 0.0, 0.0],
    }
    assert len(render_submission_source(tiny_state).encode("utf-8")) < (
        EXPORTABLE_PPO_PROFILE.max_source_bytes
    )
    try:
        ExportProfile(fcnet_hiddens=(256,)).validate()
    except ValueError as exc:
        assert "source-size" in str(exc)
    else:
        raise AssertionError("oversized export profile was not rejected")


if __name__ == "__main__":
    run_export_smoke_check()
    print("export smoke: ok")
