"""Train, export, validate, and evaluate a standalone PPO submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from python.market_game_downstream.rl.evaluate import CSV_COLUMNS
from python.market_game_downstream.rl.export.export_rllib_checkpoint import export_checkpoint
from python.market_game_downstream.rl.export.validators import validate_submission_file
from python.market_game_downstream.rl.export_profiles import EXPORTABLE_PPO_PROFILE
from python.market_game_downstream.rl.scenarios import (
    evaluate_scenario,
    format_scenario_choices,
    scenario_by_name,
    submitted_function_policy_factory,
    weekly_training_scenarios,
)
from python.market_game_downstream.rl.training.train_rllib import train


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
    """Run the full competition-oriented RL workflow."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="train an export-compatible PPO policy and write a validated submission"
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="print built-in scenario IDs and exit",
    )
    parser.add_argument("--scenario", default="week_1_baselines")
    parser.add_argument("--scenario-seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--episodes-per-iteration",
        type=int,
        default=8,
        help="24-hour market-game episodes collected per PPO iteration",
    )
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output", required=False, default="runs/rl_exportable/submission.py")
    parser.add_argument(
        "--evaluate-scenario",
        action="append",
        default=[],
        help="scenario to score the exported submission against; may repeat",
    )
    args = parser.parse_args()
    if args.list_scenarios:
        print(format_scenario_choices(seed=args.scenario_seed))
        return
    train_exportable(
        scenario=args.scenario,
        output=args.output,
        scenario_seed=args.scenario_seed,
        iterations=args.iterations,
        train_batch_size=args.episodes_per_iteration * 24,
        minibatch_size=min(args.episodes_per_iteration * 24, 64),
        checkpoint_dir=args.checkpoint_dir,
        evaluation_scenarios=args.evaluate_scenario or [args.scenario],
    )


if __name__ == "__main__":
    main()
