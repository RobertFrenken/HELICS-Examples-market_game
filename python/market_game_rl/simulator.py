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


BATTERY_CAPACITY = 20.0
BATTERY_MAX_DISCHARGE = 10.0
BATTERY_MAX_CHARGE = 5.0
INITIAL_BATTERY = 0.0
INITIAL_PRICE = 0.5
EPISODE_HOURS = 24

PROFILE1_DEMAND = [
    2,
    1,
    1,
    1,
    2,
    4,
    6,
    8,
    9,
    7,
    5,
    4,
    3,
    4,
    5,
    7,
    9,
    12,
    10,
    7,
    5,
    4,
    2,
    2,
]


class HousePolicy(Protocol):
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


@dataclass
class BatteryState:
    energy: float = INITIAL_BATTERY

    def change(self, delta: float) -> None:
        if delta < 0.0:
            discharge = abs(delta)
            if discharge > self.energy:
                raise ValueError("requested discharge exceeds current charge level")
            if discharge > BATTERY_MAX_DISCHARGE:
                raise ValueError("requested discharge exceeds maximum discharge rate")
            self.energy -= discharge
        else:
            if self.energy + delta > BATTERY_CAPACITY:
                raise ValueError("requested charge exceeds maximum capacity")
            if delta > BATTERY_MAX_CHARGE:
                raise ValueError("requested charge rate exceeds maximum rate")
            self.energy += delta


@dataclass
class SimHouse:
    policy: HousePolicy
    demand: list[float] = field(default_factory=lambda: list(PROFILE1_DEMAND))
    battery: BatteryState = field(default_factory=BatteryState)
    loads: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    battery_history: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

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


def compute_price_from_average_load(average_load: float) -> float:
    """Mirror ``market_maker.compute_new_price(total, feds)`` after averaging."""
    m = average_load
    if m < 3.0:
        return 0.1
    if m < 6.0:
        return 0.1 + 0.03 * (m - 3.0)
    if m < 9.0:
        return 0.19 + 0.1 * (m - 6.0)
    if m < 13.0:
        return 0.49 + 0.25 * (m - 9.0)
    return 1.49 + 1.0 * (m - 13.0)


def compute_price_from_total_load(total_load: float, house_count: int) -> float:
    if house_count == 0:
        return 0.1
    return compute_price_from_average_load(total_load / house_count)


def clamp_market_load(value: float, base_demand: float, battery: BatteryState) -> tuple[float, str]:
    """Mirror ``battery.ensure_valid`` and ``check_valid`` effective behavior.

    Negative market-facing load is allowed when battery discharge supports it.
    The comparison operators intentionally match the original code's inclusive
    threshold behavior.
    """
    if value >= base_demand + (BATTERY_CAPACITY - battery.energy):
        return base_demand + (BATTERY_CAPACITY - battery.energy), (
            "listed battery charge rate exceeds available battery storage capacity"
        )
    if value >= base_demand + BATTERY_MAX_CHARGE:
        return base_demand + BATTERY_MAX_CHARGE, (
            "listed battery charge rate exceeds maximum charge rate"
        )
    if value <= base_demand - battery.energy:
        return base_demand - battery.energy, "listed consumption exceeds available battery energy"
    if value <= base_demand - BATTERY_MAX_DISCHARGE:
        return base_demand - BATTERY_MAX_DISCHARGE, (
            "listed consumption exceeds max battery discharge rate"
        )
    return value, ""


def run_episode(
    policies: list[HousePolicy],
    demand_profile: list[float] | None = None,
    initial_price: float = INITIAL_PRICE,
) -> SimulationResult:
    """Run one 24-hour market-game episode."""
    demand = list(demand_profile or PROFILE1_DEMAND)
    houses = [SimHouse(policy=policy, demand=list(demand)) for policy in policies]
    current_price = initial_price
    price_history: list[float] = []
    records: list[HourRecord] = []

    for hour in range(EPISODE_HOURS):
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
            load, warning = clamp_market_load(proposed_load, base, house.battery)
            if warning:
                house.warnings.append(warning)

            house.battery.change(load - base)
            cost = current_price * load

            house.loads.append(load)
            house.costs.append(cost)
            house.battery_history.append(house.battery.energy)

            total_load += load
            loads_by_house[house.policy.name] = load
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

