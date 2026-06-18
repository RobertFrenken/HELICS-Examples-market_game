"""Checks for downstream RL action adapters."""

from __future__ import annotations

from python.market_game_downstream.rl.agents.action_spaces import (
    ContinuousNormalizedDeltaActionSpace,
    DiscreteBatteryPostureActionSpace,
    IntegerBatteryDeltaActionSpace,
)
from python.market_game_downstream.rl.envs.env import MarketGameEnv


def run_actions_check() -> None:
    posture = DiscreteBatteryPostureActionSpace()
    assert posture.market_load(-1, 12.0, 20.0) == 2.0
    assert posture.market_load(0, 12.0, 20.0) == 12.0
    assert posture.market_load(1, 2.0, 0.0) == 7.0

    integer_delta = IntegerBatteryDeltaActionSpace()
    assert integer_delta.market_load(-3, 12.0, 20.0) == 9.0
    assert integer_delta.market_load(4, 2.0, 0.0) == 6.0
    try:
        integer_delta.market_load(6, 2.0, 0.0)
    except ValueError as exc:
        assert "invalid integer battery delta" in str(exc)
    else:
        raise AssertionError("out-of-range integer delta was not rejected")

    continuous = ContinuousNormalizedDeltaActionSpace()
    assert continuous.market_load(-0.25, 12.0, 20.0) == 9.5
    assert continuous.market_load(0.5, 2.0, 0.0) == 4.5

    env = MarketGameEnv(action_space=integer_delta)
    obs, info = env.reset()
    assert info["action_space"] == "IntegerBatteryDeltaActionSpace"
    obs, reward, terminated, truncated, info = env.step(4)
    assert not terminated
    assert not truncated
    assert info["diagnostics"].own_market_load == 6.0


if __name__ == "__main__":
    run_actions_check()
    print("actions: ok")
