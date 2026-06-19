"""Smoke checks for the dependency-free market-game environment."""

from __future__ import annotations

from python.market_game_downstream.rl.observations import (
    OBSERVATION_SCHEMAS,
    ObservationMode,
)
from python.market_game_downstream.rl.adapters.policies import (
    FlattenDemandPolicy,
    PriceAwarePolicy,
)
from python.market_game_downstream.rl.agents.primitives import BatteryPosture
from python.market_game_downstream.rl.envs.env import MarketGameEnv
from python.market_game_downstream.rl.training.rewards import RewardConfig, market_game_reward


def run_env_smoke_check() -> None:
    default_env = MarketGameEnv()
    assert len({policy.name for policy in default_env.opponent_policies}) == len(
        default_env.opponent_policies
    )

    for mode, schema in OBSERVATION_SCHEMAS.items():
        assert len(schema) == len(set(schema)), mode
        env = MarketGameEnv(observation_mode=mode)
        obs, info = env.reset()
        assert len(obs) == len(schema), (mode, len(obs))
        assert info["hour"] == 0
        obs, reward, terminated, truncated, info = env.step(BatteryPosture.NEUTRAL)
        assert len(obs) == len(schema), (mode, len(obs))
        assert isinstance(reward, float)
        assert not truncated

    env = MarketGameEnv(
        opponent_policies=[FlattenDemandPolicy(), PriceAwarePolicy()],
        observation_mode=ObservationMode.PRICE_HISTORY,
    )
    env.reset()
    terminated = False
    steps = 0
    total_reward = 0.0
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(BatteryPosture.NEUTRAL)
        assert not truncated
        total_reward += reward
        steps += 1

    assert steps == 24, steps
    assert abs(total_reward + info["total_cost"]) < 1e-9
    assert market_game_reward(own_cost=3.0, final_battery=0.0, terminated=False) == -3.0
    assert market_game_reward(
        own_cost=3.0,
        final_battery=5.0,
        terminated=True,
        config=RewardConfig(final_battery_target=0.0, final_battery_penalty=2.0),
    ) == -13.0

    obs_again, info_again = env.reset()
    assert len(obs_again) == len(OBSERVATION_SCHEMAS[ObservationMode.PRICE_HISTORY])
    assert info_again["hour"] == 0
    assert info_again["total_cost"] == 0


if __name__ == "__main__":
    run_env_smoke_check()
    print("env smoke: ok")
