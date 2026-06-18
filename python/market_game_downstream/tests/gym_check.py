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
from python.market_game_downstream.rl.agents.actions import (
    ContinuousNormalizedDeltaActionSpace,
    IntegerBatteryDeltaActionSpace,
)
from python.market_game_downstream.rl.envs.gym_env import GymMarketGameEnv


def run_gym_smoke_check() -> None:
    for mode in ObservationMode:
        env = GymMarketGameEnv(
            opponent_policies=[FlattenDemandPolicy(), PriceAwarePolicy()],
            observation_mode=mode,
        )
        obs, info = env.reset(seed=1)
        assert obs.shape == (len(observation_schema(mode)),)
        assert info["hour"] == 0
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (len(observation_schema(mode)),)
        assert isinstance(reward, float)
        assert not truncated
        assert isinstance(terminated, bool)

    env = GymMarketGameEnv(observation_mode=ObservationMode.LOCAL)
    check_env(env, skip_render_check=True)
    env.reset(seed=1)
    try:
        env.step(99)
    except ValueError as exc:
        assert "invalid Gym action" in str(exc)
    else:
        raise AssertionError("invalid Gym action was not rejected")

    integer_env = GymMarketGameEnv(action_space=IntegerBatteryDeltaActionSpace())
    assert integer_env.action_space.start == -10
    assert integer_env.action_space.n == 16
    obs, info = integer_env.reset(seed=1)
    obs, reward, terminated, truncated, info = integer_env.step(4)
    assert info["diagnostics"].own_market_load == 6.0

    continuous_env = GymMarketGameEnv(action_space=ContinuousNormalizedDeltaActionSpace())
    assert continuous_env.action_space.shape == ()
    obs, info = continuous_env.reset(seed=1)
    obs, reward, terminated, truncated, info = continuous_env.step(0.5)
    assert info["diagnostics"].own_market_load == 4.5


if __name__ == "__main__":
    run_gym_smoke_check()
    print("gym env smoke: ok")
