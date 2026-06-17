"""Reward shaping for market-game RL training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Small reward contract for training without changing simulator scoring."""

    cost_weight: float = 1.0
    final_battery_target: float | None = None
    final_battery_penalty: float = 0.0


def market_game_reward(
    *,
    own_cost: float,
    final_battery: float,
    terminated: bool,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Return the per-step learner reward.

    By default this is exactly negative hourly cost. Optional terminal shaping
    can encourage a specific final battery level while leaving evaluation costs
    and exported submission behavior unchanged.
    """
    reward = -config.cost_weight * own_cost
    if terminated and config.final_battery_target is not None:
        reward -= config.final_battery_penalty * abs(
            final_battery - config.final_battery_target
        )
    return reward
