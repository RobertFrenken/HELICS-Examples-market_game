"""Generate HELICS runner configs from downstream RL scenario definitions."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import shlex
from typing import Any

from .scenarios import CompetitionScenario, load_scenarios, scenario_by_name


DEFAULT_PORT = 23404


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="build a canonical python/market_game HELICS config from an RL scenario"
    )
    parser.add_argument(
        "--scenario",
        default="week_1_baselines",
        help="named scenario to convert; defaults to week_1_baselines",
    )
    parser.add_argument(
        "--scenario-config",
        default=None,
        help="JSON scenario config path; defaults to rl/scenario_configs/weekly.json",
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=None,
        help="seed override for generated demand profiles and stochastic opponents",
    )
    parser.add_argument(
        "--submission",
        default=None,
        help="optional standalone compute_demand(...) file to add as the first house",
    )
    parser.add_argument(
        "--submission-name",
        default="SubmittedHouse",
        help="name for --submission in the HELICS federation",
    )
    parser.add_argument(
        "--output",
        default="python/market_game/houses.json",
        help="path for the generated helics run JSON",
    )
    parser.add_argument(
        "--launcher",
        choices=("uv", "plain"),
        default="uv",
        help="command prefix for generated federates",
    )
    parser.add_argument(
        "--broker-port",
        type=int,
        default=DEFAULT_PORT,
        help="local broker port for the generated federation",
    )
    return parser


def build_helics_runner(
    scenario: CompetitionScenario,
    *,
    repo_root: Path,
    launcher: str = "uv",
    broker_port: int = DEFAULT_PORT,
    submission: str | Path | None = None,
    submission_name: str = "SubmittedHouse",
) -> dict[str, Any]:
    """Return a ``helics run`` JSON object for one scenario."""
    market_game_dir = repo_root / "python" / "market_game"
    command_prefix = "uv run " if launcher == "uv" else ""
    broker = f"localhost:{broker_port}"
    house_specs = []
    if submission is not None:
        submission_path = Path(submission)
        if not submission_path.is_absolute():
            submission_path = repo_root / submission_path
        house_specs.append(
            _submission_house_spec(
                submission_path.resolve(),
                name=submission_name,
                broker=broker,
                command_prefix=command_prefix,
            )
        )
    house_specs.extend(
        _submission_player_house_spec(player, repo_root, broker, command_prefix)
        for player in scenario.submitted_players()
    )
    house_specs.extend(
        _policy_house_spec(policy, broker=broker, command_prefix=command_prefix)
        for policy in scenario.policies()
        if type(policy).__name__ != "FunctionSubmissionPolicy"
    )

    federates: list[dict[str, str]] = [
        {
            "directory": market_game_dir.as_posix(),
            "host": "localhost",
            "name": "broker",
            "exec": (
                f"{command_prefix}helics_broker -f {len(house_specs) + 1} "
                f"-t zmqss --ipv4 -p {broker_port} --loglevel=warning"
            ),
        }
    ]
    federates.extend(
        {
            "directory": market_game_dir.as_posix(),
            "host": "localhost",
            "name": spec["federate_name"],
            "exec": spec["exec"],
        }
        for spec in house_specs
    )
    federates.append(
        {
            "directory": market_game_dir.as_posix(),
            "host": "localhost",
            "name": "market_maker",
            "exec": (
                f"{command_prefix}python -u market_maker.py --auto "
                f"--broker {broker} --no-plot --profile {scenario.profile_type}"
            ),
        }
    )
    return {
        "name": f"market_game_{scenario.name}",
        "federates": federates,
    }


def write_helics_runner(runner: dict[str, Any], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(runner, indent=3) + "\n", encoding="utf-8")


def _policy_house_spec(policy: object, *, broker: str, command_prefix: str) -> dict[str, str]:
    name = str(getattr(policy, "name"))
    policy_type = type(policy).__name__
    kwargs = _policy_kwargs(policy)
    kwargs.setdefault("name", name)
    kwargs_json = json.dumps(kwargs, sort_keys=True, separators=(",", ":"))
    return {
        "federate_name": _federate_name(name),
        "exec": (
            f"{command_prefix}python -u policy_house.py --broker {broker} --no-plot "
            f"--name {shlex.quote(name)} --policy-type {shlex.quote(policy_type)} "
            f"--policy-kwargs {shlex.quote(kwargs_json)}"
        ),
    }


def _submission_house_spec(
    submission: str | Path,
    *,
    name: str,
    broker: str,
    command_prefix: str,
) -> dict[str, str]:
    return {
        "federate_name": _federate_name(name),
        "exec": (
            f"{command_prefix}python -u policy_house.py --broker {broker} --no-plot "
            f"--name {shlex.quote(name)} --submission {shlex.quote(Path(submission).as_posix())}"
        ),
    }


def _submission_player_house_spec(
    player: dict[str, Any],
    repo_root: Path,
    broker: str,
    command_prefix: str,
) -> dict[str, str]:
    path = player["path"]
    name = str(player.get("name", "SubmittedHouse"))
    submission_path = Path(str(path))
    if not submission_path.is_absolute():
        submission_path = repo_root / submission_path
    return _submission_house_spec(
        submission_path.resolve(),
        name=name,
        broker=broker,
        command_prefix=command_prefix,
    )


def _policy_kwargs(policy: object) -> dict[str, Any]:
    if not is_dataclass(policy):
        return {}
    result: dict[str, Any] = {}
    for field in fields(policy):
        if not field.init or field.name.startswith("_"):
            continue
        value = getattr(policy, field.name)
        if _is_json_scalar(value) or _is_json_list(value):
            result[field.name] = value
    return result


def _is_json_scalar(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _is_json_list(value: object) -> bool:
    return isinstance(value, list) and all(_is_json_scalar(item) for item in value)


def _federate_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "_-" else "_" for char in name)


def main() -> None:
    args = build_parser().parse_args()
    scenarios = load_scenarios(args.scenario_config, seed=args.scenario_seed)
    scenario = scenario_by_name(args.scenario, scenarios)
    runner = build_helics_runner(
        scenario,
        repo_root=Path.cwd(),
        launcher=args.launcher,
        broker_port=args.broker_port,
        submission=args.submission,
        submission_name=args.submission_name,
    )
    write_helics_runner(runner, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
