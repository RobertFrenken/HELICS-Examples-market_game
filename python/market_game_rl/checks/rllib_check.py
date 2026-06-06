"""Smoke check for Ray RLlib PPO integration."""

from __future__ import annotations

from ..agents.observations import ObservationMode
from ..training.train_rllib import train


def run_rllib_smoke_check() -> None:
    results = train(iterations=1, observation_mode=ObservationMode.LOCAL)
    assert len(results) == 1
    assert results[0]["iteration"] == 1
    assert results[0]["episode_return_mean"] is not None
    assert results[0]["episode_len_mean"] == 24.0


if __name__ == "__main__":
    run_rllib_smoke_check()
    print("rllib smoke: ok")
