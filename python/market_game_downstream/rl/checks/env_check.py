"""Smoke checks for the dependency-free market-game environment."""

from __future__ import annotations

from ..envs.env import MarketGameEnv
from ..agents.observations import ObservationMode, OBSERVATION_SCHEMAS
from ..agents.policies import FlattenDemandPolicy, PriceAwarePolicy
from ..core.rules import BatteryAction


def run_env_smoke_check() -> None:
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
