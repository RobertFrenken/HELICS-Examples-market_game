"""Minimal Ray RLlib training entry point for the market-game environment."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import ray
from ray.rllib.core.columns import Columns
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import torch

from python.market_game_downstream.core import DEFAULT_CONFIG
from ..envs.gym_env import GymMarketGameEnv
from ..agents.observations import ObservationMode
from ..agents.policies import FlattenDemandPolicy, PriceAwarePolicy
from ..scenarios import scenario_by_name, weekly_training_scenarios


ENV_NAME = "market_game_downstream.rl"


def make_default_opponents() -> list:
    """Return the fixed opponent population used by smoke training."""
    return [FlattenDemandPolicy(), PriceAwarePolicy()]


def make_env(env_config: dict[str, Any] | None = None) -> GymMarketGameEnv:
    env_config = env_config or {}
    scenario_name = env_config.get("scenario")
    opponent_policies = make_default_opponents()
    market_config = None
    scenario_observation_mode = None
    if scenario_name:
        scenarios = weekly_training_scenarios(seed=env_config.get("scenario_seed", 1))
        scenario = scenario_by_name(str(scenario_name), scenarios)
        opponent_policies, market_config = scenario.to_env_config()
    observation_mode = env_config.get(
        "observation_mode",
        scenario_observation_mode or ObservationMode.PRICE_HISTORY.value,
    )
    return GymMarketGameEnv(
        opponent_policies=opponent_policies,
        config=market_config if market_config is not None else DEFAULT_CONFIG,
        observation_mode=observation_mode,
        final_battery_target=env_config.get("final_battery_target"),
        final_battery_penalty=env_config.get("final_battery_penalty", 0.0),
    )


def build_ppo_config(
    observation_mode: ObservationMode | str | None = ObservationMode.PRICE_HISTORY,
    scenario: str | None = None,
    scenario_seed: int | None = None,
    train_batch_size: int = 192,
    minibatch_size: int = 64,
    num_epochs: int = 2,
    lr: float = 3e-4,
    gamma: float = 0.99,
    num_env_runners: int = 0,
) -> PPOConfig:
    """Build a small local PPO config for smoke training and early experiments."""
    env_config = {}
    if observation_mode is not None:
        env_config["observation_mode"] = ObservationMode(observation_mode).value
    if scenario:
        env_config["scenario"] = scenario
    if scenario_seed is not None:
        env_config["scenario_seed"] = scenario_seed
    config = (
        PPOConfig()
        .environment(
            env=ENV_NAME,
            env_config=env_config,
        )
        .framework("torch")
        .env_runners(num_env_runners=num_env_runners)
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            lr=lr,
            gamma=gamma,
        )
    )
    return config


def extract_training_summary(iteration: int, result: dict[str, Any]) -> dict[str, Any]:
    """Extract stable summary fields across RLlib result formats."""
    env_runners = result.get("env_runners") or {}
    return {
        "iteration": iteration,
        "episode_return_mean": result.get(
            "episode_reward_mean",
            env_runners.get("episode_return_mean"),
        ),
        "episode_len_mean": result.get(
            "episode_len_mean",
            env_runners.get("episode_len_mean"),
        ),
        "num_env_steps_sampled": result.get(
            "num_env_steps_sampled_lifetime",
            env_runners.get("num_env_steps_sampled_lifetime"),
        ),
    }


def evaluate_algorithm(
    algorithm: Any,
    observation_mode: ObservationMode | str | None,
    scenarios: list[str],
    scenario_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run deterministic full-episode evaluations for a trained algorithm."""
    rows = []
    for scenario in scenarios:
        env_config = {
            "scenario": scenario,
            "scenario_seed": scenario_seed,
        }
        if observation_mode is not None:
            env_config["observation_mode"] = ObservationMode(observation_mode).value
        env = make_env(env_config)
        obs, info = env.reset(seed=scenario_seed)
        terminated = False
        truncated = False
        episode_return = 0.0
        steps = 0
        while not (terminated or truncated):
            action = _compute_action(algorithm, obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_return += reward
            steps += 1
        rows.append(
            {
                "scenario": scenario,
                "episode_return": episode_return,
                "total_cost": -episode_return,
                "episode_len": steps,
                "final_battery": info["battery"],
            }
        )
    return rows


def _compute_action(algorithm: Any, obs: Any) -> int:
    """Compute one action across RLlib old/new API result formats."""
    if getattr(algorithm.config, "enable_rl_module_and_learner", False):
        module = algorithm.get_module()
        obs_batch = torch.as_tensor(np.expand_dims(obs, axis=0), dtype=torch.float32)
        outputs = module.forward_inference({Columns.OBS: obs_batch})
        if Columns.ACTIONS in outputs:
            result = outputs[Columns.ACTIONS][0]
        else:
            dist_class = module.get_inference_action_dist_cls()
            dist = dist_class.from_logits(outputs[Columns.ACTION_DIST_INPUTS])
            result = dist.to_deterministic().sample()[0]
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        if hasattr(result, "item"):
            return int(result.item())
        return int(result)

    result = algorithm.compute_single_action(obs, explore=False)
    if isinstance(result, tuple):
        result = result[0]
    if hasattr(result, "item"):
        return int(result.item())
    return int(result)


def _print_evaluation_rows(rows: list[dict[str, Any]]) -> None:
    print("eval_scenario,episode_return,total_cost,episode_len,final_battery")
    for row in rows:
        print(
            f"{row['scenario']},"
            f"{row['episode_return']:.10f},"
            f"{row['total_cost']:.10f},"
            f"{row['episode_len']},"
            f"{row['final_battery']:.10f}"
        )


def train(
    iterations: int,
    observation_mode: ObservationMode | str | None,
    scenario: str | None = None,
    scenario_seed: int | None = None,
    checkpoint_dir: str | None = None,
    evaluation_scenario_names: list[str] | None = None,
    train_batch_size: int = 192,
    minibatch_size: int = 64,
    num_epochs: int = 2,
    lr: float = 3e-4,
    gamma: float = 0.99,
    num_env_runners: int = 0,
) -> list[dict[str, Any]]:
    """Run a small PPO training job.

    This is an integration/smoke-training entry point, not a tuned experiment.
    The printed return is useful for confirming RLlib is collecting complete
    24-hour episodes, but it should not be interpreted as a meaningful learned
    policy result yet.
    """
    register_env(ENV_NAME, make_env)
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level=logging.ERROR,
        num_cpus=1,
    )
    algorithm = build_ppo_config(
        observation_mode=observation_mode,
        scenario=scenario,
        scenario_seed=scenario_seed,
        train_batch_size=train_batch_size,
        minibatch_size=minibatch_size,
        num_epochs=num_epochs,
        lr=lr,
        gamma=gamma,
        num_env_runners=num_env_runners,
    ).build_algo()
    results = []
    try:
        for index in range(iterations):
            result = algorithm.train()
            summary = extract_training_summary(index + 1, result)
            results.append(summary)
            print(
                "iteration={iteration} episode_return_mean={episode_return_mean} "
                "episode_len_mean={episode_len_mean} "
                "num_env_steps_sampled={num_env_steps_sampled}".format(**summary)
            )
        if checkpoint_dir:
            checkpoint_result = algorithm.save(Path(checkpoint_dir).as_posix())
            checkpoint = getattr(checkpoint_result, "checkpoint", checkpoint_result)
            checkpoint_path = getattr(checkpoint, "path", checkpoint_result)
            print(f"checkpoint={checkpoint_path}")
        if evaluation_scenario_names:
            rows = evaluate_algorithm(
                algorithm,
                observation_mode=observation_mode,
                scenarios=evaluation_scenario_names,
                scenario_seed=scenario_seed,
            )
            _print_evaluation_rows(rows)
    finally:
        algorithm.stop()
        ray.shutdown()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="train PPO with Ray RLlib on the market-game env")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=192)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=0,
        help="parallel RLlib env runners; keep 0 for local smoke tests",
    )
    parser.add_argument(
        "--observation-mode",
        choices=[mode.value for mode in ObservationMode],
        default=None,
        help="override scenario-declared training observation mode",
    )
    parser.add_argument(
        "--scenario",
        help="named scenario from the scenario config to train against",
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        help="seed override for generated scenario demand profiles and stochastic opponents",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="directory where RLlib should save a checkpoint after training",
    )
    parser.add_argument(
        "--evaluate-scenario",
        action="append",
        default=[],
        help="named scenario to evaluate after training; may be provided more than once",
    )
    args = parser.parse_args()
    train(
        iterations=args.iterations,
        observation_mode=args.observation_mode,
        scenario=args.scenario,
        scenario_seed=args.scenario_seed,
        checkpoint_dir=args.checkpoint_dir,
        evaluation_scenario_names=args.evaluate_scenario,
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        gamma=args.gamma,
        num_env_runners=args.num_env_runners,
    )


if __name__ == "__main__":
    main()
