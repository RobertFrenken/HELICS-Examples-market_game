"""Adapters around the official ``compute_demand(...)`` policy shape."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


ComputeDemand = Callable[[float, int, float, list[float], list[float]], float]


class ComputeDemandPolicy(Protocol):
    name: str

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        """Return this hour's market-facing load."""


def policy_to_compute_demand(policy: ComputeDemandPolicy) -> ComputeDemand:
    """Return the official callable shape for a policy object."""

    def compute_demand(
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return call_compute_demand(
            policy,
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )

    return compute_demand


def call_compute_demand(
    policy: ComputeDemandPolicy | ComputeDemand,
    price: float,
    hour: int,
    battery_charge: float,
    demand: list[float],
    price_history: list[float],
) -> float:
    """Call either a policy object or a plain compute function."""
    compute = policy if callable(policy) else policy.compute_demand
    return float(compute(price, hour, battery_charge, demand, price_history))


def reset_policy(policy: object) -> None:
    """Reset stateful policies without forcing every policy to implement reset."""
    reset = getattr(policy, "reset", None)
    if reset is not None:
        reset()
