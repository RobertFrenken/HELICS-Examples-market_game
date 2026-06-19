"""Adapters from local policy objects to the competition function surface."""

from __future__ import annotations

from dataclasses import dataclass

from ..adapters.callables import (
    ComputeDemand,
    ComputeDemandPolicy as ExportablePolicy,
    call_compute_demand,
    policy_to_compute_demand,
)


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
        return call_compute_demand(
            self.compute_fn,
            price,
            hour,
            battery_charge,
            demand,
            price_history,
        )
