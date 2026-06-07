"""Smoke checks for the dependency-free market-game environment."""

from __future__ import annotations

from python.market_game_downstream.rl.agents.observations import (
    OBSERVATION_SCHEMAS,
)
from python.market_game_downstream.rl.agents.policies import (
    FlattenDemandPolicy,
    PriceAwarePolicy,
)
from python.market_game_downstream.rl.core.rules import BatteryAction
from python.market_game_downstream.rl.envs.env import MarketGameEnv


def run_env_smoke_check() -> None:
    default_env = MarketGameEnv()
    assert len({policy.name for policy in default_env.opponent_policies}) == len(
        default_env.opponent_policies
    )

    for mode, schema in OBSERVATION_SCHEMAS.items():
        expected_dim = len(schema)
        assert expected_dim == len(set(schema)), mode
        env = MarketGameEnv(
            opponent_policies=[FlattenDemandPolicy(), PriceAwarePolicy()],
            observation_mode=mode,
        )
        obs, info = env.reset()
        assert len(obs) == expected_dim, (mode, len(obs))
        assert info["hour"] == 0

        terminated = False
        steps = 0
        total_reward = 0.0
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(BatteryAction.NEUTRAL)
            assert not truncated
            assert len(obs) == expected_dim, (mode, len(obs))
            total_reward += reward
            steps += 1

        assert steps == 24, steps
        assert abs(total_reward + info["total_cost"]) < 1e-9

        obs_again, info_again = env.reset()
        assert len(obs_again) == expected_dim
        assert info_again["hour"] == 0
        assert info_again["total_cost"] == 0


if __name__ == "__main__":
    run_env_smoke_check()
    print("env smoke: ok")
