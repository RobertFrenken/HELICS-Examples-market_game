"""Smoke check for Ray RLlib PPO integration."""

from __future__ import annotations

from pathlib import Path
import tempfile
from unittest.mock import patch

from python.market_game_downstream.rl.agents.observations import ObservationMode
from python.market_game_downstream.rl.export.validators import MAX_SOURCE_BYTES, validate_submission_file
from python.market_game_downstream.rl.envs.scenarios import scenario_by_name, weekly_training_scenarios
from python.market_game_downstream.rl.evaluate import (
    evaluate_scenario,
    submitted_function_policy_factory,
)
from python.market_game_downstream.rl.training.rllib import train
from python.market_game_downstream.rl.training.train import main as train_cli_main


def run_rllib_smoke_check() -> None:
    results = train(iterations=1, observation_mode=ObservationMode.LOCAL)
    assert len(results) == 1
    assert results[0]["iteration"] == 1
    assert results[0]["episode_return_mean"] is not None
    assert results[0]["episode_len_mean"] == 24.0


def run_rllib_exportable_check() -> None:
    """Train one tiny PPO iteration and verify the exported submission contract."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_path = temp_path / "submission.py"
        config_path = temp_path / "train.toml"
        config_path.write_text(
            "\n".join(
                [
                    "[training]",
                    'mode = "exportable"',
                    'scenario = "week_1_baselines"',
                    "scenario_seed = 1",
                    "iterations = 1",
                    "episodes_per_iteration = 1",
                    "minibatch_size = 24",
                    f'checkpoint_dir = "{(temp_path / "checkpoint").as_posix()}"',
                    f'output = "{output_path.as_posix()}"',
                    'evaluate_scenario = ["week_1_baselines"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        with patch(
            "sys.argv",
            [
                "train",
                "--config",
                str(config_path),
            ],
        ):
            train_cli_main()
        assert output_path.stat().st_size < MAX_SOURCE_BYTES
        report = validate_submission_file(output_path)
        assert report.hours == 24
        assert report.clamps == 0
        assert not report.boundary_warnings

        scenarios = weekly_training_scenarios(seed=1)
        scenario = scenario_by_name("week_1_baselines", scenarios).with_policy_factory(
            submitted_function_policy_factory(output_path),
            first=True,
        )
        rows = [
            row
            for row in evaluate_scenario(scenario)
            if row["agent"] == "SubmittedHouse"
        ]
        assert len(rows) == 1
        assert rows[0]["clamps"] == "0"
        assert rows[0]["boundary_warnings"] == "0"


if __name__ == "__main__":
    run_rllib_smoke_check()
    print("rllib smoke: ok")
    run_rllib_exportable_check()
    print("rllib exportable e2e: ok")
