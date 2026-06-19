"""Checks for downstream RL action adapters."""

from __future__ import annotations

from python.market_game_downstream.rl.envs.action_spaces import (
    ContinuousNormalizedDeltaActionSpace,
    DiscreteBatteryPostureActionSpace,
    IntegerBatteryDeltaActionSpace,
)
from python.market_game_downstream.rl.envs.action_spaces import LearnerActionSpace
from python.market_game_downstream.rl.agents.primitives import MarketPercept
from python.market_game_downstream.rl.agents.projectors import project_market_load
from python.market_game_downstream.rl.envs.env import MarketGameEnv


def run_actions_check() -> None:
    posture = DiscreteBatteryPostureActionSpace()
    assert project(posture, -1, 12.0, 20.0) == 2.0
    assert project(posture, 0, 12.0, 20.0) == 12.0
    assert project(posture, 1, 2.0, 0.0) == 7.0

    integer_delta = IntegerBatteryDeltaActionSpace()
    assert project(integer_delta, -3, 12.0, 20.0) == 9.0
    assert project(integer_delta, 4, 2.0, 0.0) == 6.0
    try:
        project(integer_delta, 6, 2.0, 0.0)
    except ValueError as exc:
        assert "invalid integer battery delta" in str(exc)
    else:
        raise AssertionError("out-of-range integer delta was not rejected")

    continuous = ContinuousNormalizedDeltaActionSpace()
    assert project(continuous, -0.25, 12.0, 20.0) == 9.5
    assert project(continuous, 0.5, 2.0, 0.0) == 4.5

    env = MarketGameEnv(action_space=integer_delta)
    obs, info = env.reset()
    assert info["action_space"] == "IntegerBatteryDeltaActionSpace"
    obs, reward, terminated, truncated, info = env.step(4)
    assert not terminated
    assert not truncated
    assert info["diagnostics"].own_market_load == 6.0


def project(
    action_space: LearnerActionSpace,
    action: object,
    base_demand: float,
    battery_charge: float,
) -> float:
    percept = MarketPercept(
        price=0.25,
        hour=0,
        battery_charge=battery_charge,
        demand=[base_demand] * 24,
        price_history=[0.25],
    )
    return project_market_load(action_space.decode(action, percept), percept)


if __name__ == "__main__":
    run_actions_check()
    print("actions: ok")
