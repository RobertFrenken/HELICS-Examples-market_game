"""Optional Gymnasium adapter for ``MarketGameEnv``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from python.market_game_downstream.core import DEFAULT_CONFIG, MarketGameConfig
from .env import MarketGameEnv, default_opponent_policies
from ..training.rewards import RewardConfig
from ..agents.action_spaces import (
    DEFAULT_ACTION_SPACE,
    ContinuousNormalizedDeltaActionSpace,
    DiscreteBatteryPostureActionSpace,
    IntegerBatteryDeltaActionSpace,
)
from ..agents.interfaces import LearnerActionSpace
from ..agents.observations import ObservationMode, observation_schema
from python.market_game_downstream.core import HousePolicy


def observation_bounds(mode: ObservationMode, config: MarketGameConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative finite bounds for the selected observation schema.

    These bounds are for Gymnasium/RL library compatibility, not strict claims
    about every physically possible value in every future randomized scenario.
    """
    low: list[float] = []
    high: list[float] = []
    for name in observation_schema(mode):
        if name in {"hour_sin", "hour_cos"}:
            low.append(-1.0)
            high.append(1.0)
        elif "price" in name:
            low.append(-10.0)
            high.append(50.0)
        elif "battery" in name:
            low.append(0.0)
            high.append(config.battery_capacity)
        elif "demand" in name or "market_load" in name or "others_avg_load" in name:
            low.append(-config.max_discharge)
            high.append(max(config.demand_profile) + config.max_charge)
        elif name == "inferred_crowd_delta_lag_1":
            low.append(-config.max_discharge)
            high.append(config.max_charge)
        elif name == "tier_distance":
            low.append(0.0)
            high.append(100.0)
        elif name == "inverse_uncertainty":
            low.append(0.0)
            high.append(100.0)
        else:
            low.append(-100.0)
            high.append(100.0)
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


class GymMarketGameEnv(gym.Env):
    """Gymnasium-compatible wrapper around the dependency-free environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_policies: Sequence[HousePolicy] | None = None,
        config: MarketGameConfig = DEFAULT_CONFIG,
        observation_mode: ObservationMode | str = ObservationMode.PRICE_HISTORY,
        action_space: LearnerActionSpace = DEFAULT_ACTION_SPACE,
        reward_config: RewardConfig | None = None,
        final_battery_target: float | None = None,
        final_battery_penalty: float = 0.0,
    ):
        super().__init__()
        self.observation_mode = ObservationMode(observation_mode)
        self.action_mapper = action_space
        self.env = MarketGameEnv(
            opponent_policies=list(opponent_policies)
            if opponent_policies is not None
            else default_opponent_policies(),
            config=config,
            observation_mode=self.observation_mode,
            action_space=action_space,
            reward_config=reward_config,
            final_battery_target=final_battery_target,
            final_battery_penalty=final_battery_penalty,
        )

        obs_dim = len(observation_schema(self.observation_mode))
        low, high = observation_bounds(self.observation_mode, config)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym_action_space(action_space)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        return self._as_observation(obs), info

    def step(
        self,
        action: object,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        learner_action = _normalize_gym_action(action, self.action_mapper)
        obs, reward, terminated, truncated, info = self.env.step(learner_action)
        return self._as_observation(obs), float(reward), terminated, truncated, info

    def _as_observation(self, obs: list[float]) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)


def gym_action_space(learner_action_space: LearnerActionSpace) -> spaces.Space:
    if isinstance(learner_action_space, DiscreteBatteryPostureActionSpace):
        return spaces.Discrete(3)
    if isinstance(learner_action_space, IntegerBatteryDeltaActionSpace):
        return spaces.Discrete(
            learner_action_space.max_delta - learner_action_space.min_delta + 1,
            start=learner_action_space.min_delta,
        )
    if isinstance(learner_action_space, ContinuousNormalizedDeltaActionSpace):
        return spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32)
    raise TypeError(
        f"unsupported Gym learner action space {type(learner_action_space).__name__}"
    )


def _normalize_gym_action(
    action: object,
    learner_action_space: LearnerActionSpace,
) -> object:
    if isinstance(learner_action_space, ContinuousNormalizedDeltaActionSpace):
        return float(np.asarray(action, dtype=np.float32).item())
    try:
        action_value = int(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Gym action {action!r}") from exc
    if isinstance(learner_action_space, DiscreteBatteryPostureActionSpace):
        if action_value not in (0, 1, 2):
            raise ValueError(f"invalid Gym action {action!r}; expected 0, 1, or 2")
        return action_value - 1
    return action_value
