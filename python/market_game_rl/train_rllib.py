"""Minimal Ray RLlib training entry point for the market-game environment."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from .gym_env import GymMarketGameEnv
from .observations import ObservationMode
from .policies import FlattenDemandPolicy, PriceAwarePolicy


ENV_NAME = "market_game_rl"


def make_default_opponents() -> list:
    """Return the fixed opponent population used by smoke training."""
    return [FlattenDemandPolicy(), PriceAwarePolicy()]


def make_env(env_config: dict[str, Any] | None = None) -> GymMarketGameEnv:
    env_config = env_config or {}
    observation_mode = env_config.get("observation_mode", ObservationMode.PRICE_HISTORY.value)
    return GymMarketGameEnv(
        opponent_policies=make_default_opponents(),
        observation_mode=observation_mode,
        final_battery_target=env_config.get("final_battery_target"),
        final_battery_penalty=env_config.get("final_battery_penalty", 0.0),
    )


def build_ppo_config(
    observation_mode: ObservationMode | str = ObservationMode.PRICE_HISTORY,
    train_batch_size: int = 192,
) -> PPOConfig:
    """Build a small local PPO config for smoke training and early experiments."""
    config = (
        PPOConfig()
        .environment(
            env=ENV_NAME,
            env_config={"observation_mode": ObservationMode(observation_mode).value},
        )
        .framework("torch")
        .env_runners(num_env_runners=0)
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=64,
            num_epochs=2,
            lr=3e-4,
            gamma=0.99,
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


def train(iterations: int, observation_mode: ObservationMode | str) -> list[dict[str, Any]]:
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
    algorithm = build_ppo_config(observation_mode=observation_mode).build_algo()
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
    finally:
        algorithm.stop()
        ray.shutdown()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="train PPO with Ray RLlib on the market-game env")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--observation-mode",
        choices=[mode.value for mode in ObservationMode],
        default=ObservationMode.PRICE_HISTORY.value,
    )
    args = parser.parse_args()
    train(iterations=args.iterations, observation_mode=args.observation_mode)


if __name__ == "__main__":
    main()
