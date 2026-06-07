"""Smoke checks for the optional Gymnasium adapter."""

from __future__ import annotations

from gymnasium.utils.env_checker import check_env

from python.market_game_downstream.rl.agents.observations import (
    ObservationMode,
    observation_schema,
)
from python.market_game_downstream.rl.agents.policies import (
    FlattenDemandPolicy,
    PriceAwarePolicy,
)
from python.market_game_downstream.rl.envs.gym_env import GymMarketGameEnv


def run_gym_smoke_check() -> None:
    for mode in ObservationMode:
        env = GymMarketGameEnv(
            opponent_policies=[FlattenDemandPolicy(), PriceAwarePolicy()],
            observation_mode=mode,
        )
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=1)
        assert obs.shape == (len(observation_schema(mode)),)
        assert info["hour"] == 0
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (len(observation_schema(mode)),)
        assert isinstance(reward, float)
        assert not truncated
        assert isinstance(terminated, bool)
        try:
            env.step(99)
        except ValueError as exc:
            assert "invalid Gym action" in str(exc)
        else:
            raise AssertionError("invalid Gym action was not rejected")


if __name__ == "__main__":
    run_gym_smoke_check()
    print("gym env smoke: ok")
