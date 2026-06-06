"""Optional Gymnasium adapter for ``MarketGameEnv``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import DEFAULT_CONFIG, MarketGameConfig
from .env import MarketGameEnv
from .observations import ObservationMode, observation_schema
from .policies import FollowDemandPolicy
from .rules import BatteryAction
from .simulator import HousePolicy


GYM_ACTION_TO_BATTERY_ACTION = {
    0: BatteryAction.DISCHARGE,
    1: BatteryAction.NEUTRAL,
    2: BatteryAction.CHARGE,
}


def observation_bounds(mode: ObservationMode, config: MarketGameConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative finite bounds for the selected observation schema."""
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
        final_battery_target: float | None = None,
        final_battery_penalty: float = 0.0,
    ):
        super().__init__()
        self.observation_mode = ObservationMode(observation_mode)
        self.env = MarketGameEnv(
            opponent_policies=list(opponent_policies)
            if opponent_policies is not None
            else [FollowDemandPolicy(), FollowDemandPolicy()],
            config=config,
            observation_mode=self.observation_mode,
            final_battery_target=final_battery_target,
            final_battery_penalty=final_battery_penalty,
        )

        obs_dim = len(observation_schema(self.observation_mode))
        low, high = observation_bounds(self.observation_mode, config)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

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
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        battery_action = GYM_ACTION_TO_BATTERY_ACTION[int(action)]
        obs, reward, terminated, truncated, info = self.env.step(battery_action)
        return self._as_observation(obs), float(reward), terminated, truncated, info

    def _as_observation(self, obs: list[float]) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)
