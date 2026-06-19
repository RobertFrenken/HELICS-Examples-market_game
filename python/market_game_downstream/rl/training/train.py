"""Single RL training CLI for smoke, experiment, and exportable workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib
from typing import Any

from python.market_game_downstream.rl.observations import ObservationMode
from python.market_game_downstream.rl.envs.scenarios import format_scenario_choices
from python.market_game_downstream.rl.evaluate import (
    CSV_COLUMNS,
    evaluate_scenario,
    scenario_by_name,
    submitted_function_policy_factory,
    weekly_training_scenarios,
)
from python.market_game_downstream.rl.export.export_rllib_checkpoint import export_checkpoint
from python.market_game_downstream.rl.export.profiles import EXPORTABLE_PPO_PROFILE
from python.market_game_downstream.rl.export.validators import validate_submission_file
from python.market_game_downstream.rl.training.rllib import (
    _parse_fcnet_hiddens,
    train,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "smoke",
    "scenario": "week_1_baselines",
    "scenario_seed": 1,
    "iterations": None,
    "episodes_per_iteration": None,
    "checkpoint_dir": None,
    "output": "runs/rl_exportable/submission.py",
    "train_batch_size": 192,
    "minibatch_size": 64,
    "num_epochs": 2,
    "lr": 3e-4,
    "gamma": 0.99,
    "reward_cost_weight": 1.0,
    "final_battery_target": None,
    "final_battery_penalty": 0.0,
    "num_env_runners": 0,
    "observation_mode": None,
    "fcnet_hiddens": None,
    "evaluate_scenario": [],
    "list_scenarios": False,
}
CONFIG_KEYS = frozenset(DEFAULT_CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser(description="train PPO for the market-game RL workflow")
    parser.add_argument(
        "--config",
        help="TOML config file; command-line flags override config values",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "experiment", "exportable"),
        default=argparse.SUPPRESS,
        help="training workflow to run",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        default=argparse.SUPPRESS,
        help="print built-in scenario IDs and exit",
    )
    parser.add_argument("--scenario", default=argparse.SUPPRESS)
    parser.add_argument("--scenario-seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--iterations", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--episodes-per-iteration", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-dir", default=argparse.SUPPRESS)
    parser.add_argument("--output", default=argparse.SUPPRESS)
    parser.add_argument("--train-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--minibatch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--reward-cost-weight", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--final-battery-target", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--final-battery-penalty", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=argparse.SUPPRESS,
        help="parallel RLlib env runners; keep 0 for local smoke tests",
    )
    parser.add_argument(
        "--observation-mode",
        choices=[mode.value for mode in ObservationMode],
        default=argparse.SUPPRESS,
        help="observation mode for smoke/experiment training",
    )
    parser.add_argument(
        "--fcnet-hiddens",
        default=argparse.SUPPRESS,
        help="comma-separated hidden layer sizes for experiment training",
    )
    parser.add_argument(
        "--evaluate-scenario",
        action="append",
        default=argparse.SUPPRESS,
        help="scenario to evaluate after training; may be provided more than once",
    )
    parsed_args = parser.parse_args()
    args = _merge_args(parsed_args)

    if args.list_scenarios:
        print(format_scenario_choices(seed=args.scenario_seed))
        return
    if args.mode == "exportable":
        _run_exportable(args)
    else:
        _run_rllib(args)


def _run_exportable(args: argparse.Namespace) -> None:
    train_exportable(
        scenario=args.scenario,
        output=args.output,
        scenario_seed=args.scenario_seed,
        iterations=args.iterations if args.iterations is not None else 8,
        train_batch_size=_train_batch_size(args),
        minibatch_size=args.minibatch_size,
        reward_cost_weight=args.reward_cost_weight,
        final_battery_target=args.final_battery_target,
        final_battery_penalty=args.final_battery_penalty,
        checkpoint_dir=args.checkpoint_dir,
        evaluation_scenarios=args.evaluate_scenario or [args.scenario],
    )


def train_exportable(
    *,
    scenario: str,
    output: str | Path,
    scenario_seed: int = 1,
    iterations: int = 8,
    train_batch_size: int = 192,
    minibatch_size: int = 64,
    reward_cost_weight: float = 1.0,
    final_battery_target: float | None = None,
    final_battery_penalty: float = 0.0,
    checkpoint_dir: str | Path | None = None,
    evaluation_scenarios: list[str] | None = None,
) -> Path:
    """Train, export, validate, and evaluate a competition-compatible PPO policy."""
    profile = EXPORTABLE_PPO_PROFILE
    profile.validate()
    output_path = Path(output).resolve()
    checkpoint_path = (
        Path(checkpoint_dir).resolve()
        if checkpoint_dir is not None
        else output_path.parent / f"{output_path.stem}_checkpoint"
    )
    evaluation_scenarios = evaluation_scenarios or [scenario]

    train(
        iterations=iterations,
        observation_mode=profile.observation_mode,
        scenario=scenario,
        scenario_seed=scenario_seed,
        checkpoint_dir=checkpoint_path.as_posix(),
        evaluation_scenario_names=evaluation_scenarios,
        fcnet_hiddens=list(profile.fcnet_hiddens),
        train_batch_size=train_batch_size,
        minibatch_size=minibatch_size,
        reward_cost_weight=reward_cost_weight,
        final_battery_target=final_battery_target,
        final_battery_penalty=final_battery_penalty,
    )
    export_checkpoint(
        checkpoint_path,
        output_path,
        scenario=scenario,
        scenario_seed=scenario_seed,
        profile=profile,
    )
    report = validate_submission_file(output_path)
    print(
        "validation="
        f"hours:{report.hours},clamps:{report.clamps},"
        f"boundary_warnings:{len(report.boundary_warnings)},"
        f"source_bytes:{output_path.stat().st_size}"
    )
    _print_submission_evaluation(output_path, evaluation_scenarios, scenario_seed)
    print(f"submission={output_path}")
    print(f"checkpoint={checkpoint_path}")
    return output_path


def _print_submission_evaluation(
    output_path: Path,
    scenario_names: list[str],
    seed: int,
) -> None:
    scenarios = weekly_training_scenarios(seed=seed)
    factory = submitted_function_policy_factory(output_path)
    print(",".join(CSV_COLUMNS))
    for name in scenario_names:
        scenario = scenario_by_name(name, scenarios).with_policy_factory(factory, first=True)
        for row in evaluate_scenario(scenario):
            if row["agent"] == "SubmittedHouse":
                print(",".join(row[column] for column in CSV_COLUMNS))


def _run_rllib(args: argparse.Namespace) -> None:
    if args.output != "runs/rl_exportable/submission.py":
        raise SystemExit("--output is only valid with --mode exportable")
    observation_mode = args.observation_mode
    fcnet_hiddens = _parse_fcnet_hiddens_arg(args.fcnet_hiddens)
    if args.mode == "smoke":
        observation_mode = observation_mode or ObservationMode.LOCAL.value
        fcnet_hiddens = fcnet_hiddens or [8]
    else:
        observation_mode = observation_mode or ObservationMode.PRICE_HISTORY.value
    train(
        iterations=args.iterations if args.iterations is not None else 1,
        observation_mode=observation_mode,
        scenario=args.scenario,
        scenario_seed=args.scenario_seed,
        checkpoint_dir=args.checkpoint_dir,
        evaluation_scenario_names=args.evaluate_scenario,
        fcnet_hiddens=fcnet_hiddens,
        train_batch_size=_train_batch_size(args),
        minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        gamma=args.gamma,
        num_env_runners=args.num_env_runners,
        reward_cost_weight=args.reward_cost_weight,
        final_battery_target=args.final_battery_target,
        final_battery_penalty=args.final_battery_penalty,
    )


def _merge_args(parsed_args: argparse.Namespace) -> argparse.Namespace:
    cli_values = vars(parsed_args)
    config_path = cli_values.pop("config", None)
    values = dict(DEFAULT_CONFIG)
    if config_path:
        values.update(_load_config(config_path))
    values.update(cli_values)
    _validate_values(values)
    return argparse.Namespace(**values)


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            raw_config = tomllib.load(config_file)
    except OSError as exc:
        raise SystemExit(f"cannot read config file {config_path}: {exc}") from exc
    config = raw_config.get("training", raw_config)
    if not isinstance(config, dict):
        raise SystemExit("training config must be a TOML table")
    unknown_keys = sorted(set(config) - CONFIG_KEYS)
    if unknown_keys:
        raise SystemExit(f"unknown training config key(s): {', '.join(unknown_keys)}")
    return dict(config)


def _validate_values(values: dict[str, Any]) -> None:
    if values["mode"] not in {"smoke", "experiment", "exportable"}:
        raise SystemExit("mode must be smoke, experiment, or exportable")
    if values["episodes_per_iteration"] is not None:
        if int(values["episodes_per_iteration"]) < 1:
            raise SystemExit("episodes_per_iteration must be a positive integer")
        values["episodes_per_iteration"] = int(values["episodes_per_iteration"])
    if values["observation_mode"] is not None:
        try:
            ObservationMode(values["observation_mode"])
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in ObservationMode)
            raise SystemExit(f"observation_mode must be one of: {choices}") from exc
    if values["evaluate_scenario"] is None:
        values["evaluate_scenario"] = []
    if isinstance(values["evaluate_scenario"], str):
        values["evaluate_scenario"] = [values["evaluate_scenario"]]
    if not isinstance(values["evaluate_scenario"], list) or not all(
        isinstance(name, str) for name in values["evaluate_scenario"]
    ):
        raise SystemExit("evaluate_scenario must be a string or list of strings")


def _train_batch_size(args: argparse.Namespace) -> int:
    if args.episodes_per_iteration is None:
        return args.train_batch_size
    return args.episodes_per_iteration * 24


def _parse_fcnet_hiddens_arg(value: object) -> list[int] | None:
    if isinstance(value, list):
        hiddens = [int(item) for item in value]
        if not hiddens or any(hidden < 1 for hidden in hiddens):
            raise SystemExit("fcnet_hiddens must contain positive integers")
        return hiddens
    if value is None:
        return None
    return _parse_fcnet_hiddens(str(value))


if __name__ == "__main__":
    main()
