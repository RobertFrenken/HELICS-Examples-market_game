"""Smoke checks for the dependency-free market-game environment."""

from __future__ import annotations

from .env import MarketGameEnv
from .observations import ObservationMode
from .policies import FlattenDemandPolicy, PriceAwarePolicy
from .rules import BatteryAction


EXPECTED_OBS_DIMS = {
    ObservationMode.LOCAL: 5,
    ObservationMode.PRICE_HISTORY: 10,
    ObservationMode.INFERENCE: 16,
}


def run_env_smoke_check() -> None:
    for mode, expected_dim in EXPECTED_OBS_DIMS.items():
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
