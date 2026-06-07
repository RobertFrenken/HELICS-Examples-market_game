"""Adapters from local policy objects to the competition function surface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


ComputeDemand = Callable[[float, int, float, list[float], list[float]], float]


class ExportablePolicy(Protocol):
    name: str

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        """Return this hour's submitted market load."""


def policy_to_compute_demand(policy: ExportablePolicy) -> ComputeDemand:
    """Return a plain callable with the official ``compute_demand`` signature."""

    def compute_demand(
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return float(
            policy.compute_demand(
                price,
                hour,
                battery_charge,
                demand,
                price_history,
            )
        )

    return compute_demand


@dataclass
class FunctionSubmissionPolicy:
    """Simulator policy wrapper around a submitted ``compute_demand`` function."""

    compute_fn: ComputeDemand
    name: str = "SubmittedHouse"

    def reset(self) -> None:
        pass

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return float(
            self.compute_fn(
                price,
                hour,
                battery_charge,
                demand,
                price_history,
            )
        )
