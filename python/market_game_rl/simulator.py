"""Pure-Python simulator for the HELICS market game.

This module intentionally does not import from ``python/market_game``. Several
of those files create HELICS federates or parse CLI arguments at import time.
The rules below mirror the executable behavior inspected in:

- ``python/market_game/market_maker.py``
- ``python/market_game/house_template.py``
- ``python/market_game/battery.py``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .config import DEFAULT_CONFIG, MarketGameConfig
from .rules import clamp_market_load, compute_price_from_average_load


class HousePolicy(Protocol):
    name: str

    def reset(self) -> None:
        """Reset any episode-local policy state."""

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        """Return this hour's market-facing load."""


@dataclass
class BatteryState:
    energy: float
    config: MarketGameConfig = DEFAULT_CONFIG

    def change(self, delta: float) -> None:
        if delta < 0.0:
            discharge = abs(delta)
            if discharge > self.energy:
                raise ValueError("requested discharge exceeds current charge level")
            if discharge > self.config.max_discharge:
                raise ValueError("requested discharge exceeds maximum discharge rate")
            self.energy -= discharge
        else:
            if self.energy + delta > self.config.battery_capacity:
                raise ValueError("requested charge exceeds maximum capacity")
            if delta > self.config.max_charge:
                raise ValueError("requested charge rate exceeds maximum rate")
            self.energy += delta


@dataclass
class SimHouse:
    policy: HousePolicy
    demand: list[float]
    battery: BatteryState
    loads: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    battery_history: list[float] = field(default_factory=list)
    boundary_warnings: list[str] = field(default_factory=list)
    clamps: int = 0

    @property
    def total_cost(self) -> float:
        return sum(self.costs)

    @property
    def total_load(self) -> float:
        return sum(self.loads)


@dataclass
class HourRecord:
    hour: int
    price: float
    total_load: float
    average_load: float
    next_price: float
    loads_by_house: dict[str, float]
    batteries_by_house: dict[str, float]
    costs_by_house: dict[str, float]


@dataclass
class SimulationResult:
    houses: list[SimHouse]
    records: list[HourRecord]
    price_history: list[float]

    def total_costs(self) -> dict[str, float]:
        return {house.policy.name: house.total_cost for house in self.houses}

    def total_loads(self) -> dict[str, float]:
        return {house.policy.name: house.total_load for house in self.houses}


def run_episode(
    policies: list[HousePolicy],
    demand_profile: list[float] | None = None,
    initial_price: float | None = None,
    config: MarketGameConfig = DEFAULT_CONFIG,
) -> SimulationResult:
    """Run one 24-hour market-game episode."""
    demand = list(demand_profile or config.demand_profile)
    current_price = config.initial_price if initial_price is None else initial_price
    houses = []
    for policy in policies:
        reset = getattr(policy, "reset", None)
        if reset is not None:
            reset()
        houses.append(
            SimHouse(
                policy=policy,
                demand=list(demand),
                battery=BatteryState(config.initial_battery, config),
            )
        )
    price_history: list[float] = []
    records: list[HourRecord] = []

    for hour in range(config.episode_hours):
        price_history.append(current_price)
        total_load = 0.0
        loads_by_house: dict[str, float] = {}
        batteries_by_house: dict[str, float] = {}
        costs_by_house: dict[str, float] = {}

        for house in houses:
            base = house.demand[hour]
            proposed_load = house.policy.compute_demand(
                current_price,
                hour,
                house.battery.energy,
                house.demand,
                price_history,
            )
            clamp = clamp_market_load(proposed_load, base, house.battery.energy, config)
            market_load = clamp.market_load
            if clamp.warning:
                house.boundary_warnings.append(clamp.warning)
            if market_load != proposed_load:
                house.clamps += 1

            house.battery.change(market_load - base)
            cost = current_price * market_load

            house.loads.append(market_load)
            house.costs.append(cost)
            house.battery_history.append(house.battery.energy)

            total_load += market_load
            loads_by_house[house.policy.name] = market_load
            batteries_by_house[house.policy.name] = house.battery.energy
            costs_by_house[house.policy.name] = cost

        average_load = total_load / len(houses) if houses else 0.0
        next_price = compute_price_from_average_load(average_load) if houses else 0.1
        records.append(
            HourRecord(
                hour=hour,
                price=current_price,
                total_load=total_load,
                average_load=average_load,
                next_price=next_price,
                loads_by_house=loads_by_house,
                batteries_by_house=batteries_by_house,
                costs_by_house=costs_by_house,
            )
        )
        current_price = next_price

    return SimulationResult(houses=houses, records=records, price_history=price_history)
