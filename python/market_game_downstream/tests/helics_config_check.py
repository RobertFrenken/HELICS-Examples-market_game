"""Checks for generated HELICS runner configs."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from python.market_game_downstream.rl.helics_config import (
    build_helics_runner,
    write_helics_runner,
)
from python.market_game_downstream.rl.scenarios import (
    load_scenarios,
    scenario_by_name,
    weekly_training_scenarios,
)


def run_helics_config_check() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    scenario = scenario_by_name("week_1_baselines", weekly_training_scenarios(seed=3))
    submission = repo_root / "python" / "market_game_downstream" / "rl" / "export" / "example_threshold_submission.py"
    runner = build_helics_runner(
        scenario,
        repo_root=repo_root,
        launcher="plain",
        broker_port=23555,
        submission=submission,
        submission_name="ExportedPPOStudent",
    )

    federates = runner["federates"]
    assert runner["name"] == "market_game_week_1_baselines"
    assert len(federates) == 6
    assert federates[0]["name"] == "broker"
    assert "-f 5" in federates[0]["exec"]
    assert "-p 23555" in federates[0]["exec"]
    assert federates[-1]["name"] == "market_maker"
    assert "--profile profile1" in federates[-1]["exec"]
    assert federates[1]["name"] == "ExportedPPOStudent"
    assert "--submission" in federates[1]["exec"]
    assert submission.as_posix() in federates[1]["exec"]
    assert "policy_house.py" in federates[2]["exec"]
    assert "--policy-type FlattenDemandPolicy" in federates[2]["exec"]
    assert "--policy-kwargs" in federates[2]["exec"]

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "houses.json"
        write_helics_runner(runner, output)
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert saved == runner

        config = Path(temp_dir) / "players.json"
        config.write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "name": "players_to_helics",
                            "profile_type": "profile1",
                            "players": [
                                {
                                    "role": "submitted_function",
                                    "path": submission.as_posix(),
                                    "name": "ConfiguredSubmission",
                                }
                            ],
                            "opponents": ["PriceAwarePolicy"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        configured = load_scenarios(config)[0]
        configured_runner = build_helics_runner(
            configured,
            repo_root=repo_root,
            launcher="plain",
        )
        names = [fed["name"] for fed in configured_runner["federates"]]
        assert names == [
            "broker",
            "ConfiguredSubmission",
            "PriceAwareHouse",
            "market_maker",
        ]
        assert "--submission" in configured_runner["federates"][1]["exec"]


if __name__ == "__main__":
    run_helics_config_check()
    print("helics config: ok")
