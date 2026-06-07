"""Pure-Python simulator for the market game."""

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

    def current_charge(self) -> float:
        return self.energy

    def change(self, delta: float) -> float:
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
        return self.energy


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


@dataclass(frozen=True)
class MarketScenario:
    """Policies and rules for one repeatable market-game scenario."""

    policies: list[HousePolicy]
    demand_profile: list[float] | None = None
    initial_price: float | None = None
    config: MarketGameConfig = DEFAULT_CONFIG


@dataclass(frozen=True)
class HouseHourInput:
    """One house's proposed market load for a single hour."""

    name: str
    proposed_market_load: float
    base_demand: float
    battery: BatteryState


@dataclass(frozen=True)
class HouseHourResult:
    """Effective result for one house after validation and accounting."""

    name: str
    proposed_market_load: float
    market_load: float
    base_demand: float
    battery_charge: float
    cost: float
    warning: str = ""


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
    proposed_loads_by_house: dict[str, float] = field(default_factory=dict)
    warnings_by_house: dict[str, str] = field(default_factory=dict)


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
    scenario = MarketScenario(
        policies=policies,
        demand_profile=demand_profile,
        initial_price=initial_price,
        config=config,
    )
    return run_scenario(scenario)


def run_scenario(scenario: MarketScenario) -> SimulationResult:
    """Run one complete market-game scenario."""
    config = scenario.config
    demand = list(scenario.demand_profile or config.demand_profile)
    current_price = config.initial_price if scenario.initial_price is None else scenario.initial_price
    houses = []
    for policy in scenario.policies:
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
        hour_inputs: list[HouseHourInput] = []
        for house in houses:
            base_demand = house.demand[hour]
            proposed_load = house.policy.compute_demand(
                current_price,
                hour,
                house.battery.energy,
                house.demand,
                price_history,
            )
            hour_inputs.append(
                HouseHourInput(
                    name=house.policy.name,
                    proposed_market_load=proposed_load,
                    base_demand=base_demand,
                    battery=house.battery,
                )
            )

        record, house_results = step_market_hour(
            hour=hour,
            price=current_price,
            house_inputs=hour_inputs,
            config=config,
        )
        for house, result in zip(houses, house_results):
            if result.warning:
                house.boundary_warnings.append(result.warning)
            if result.market_load != result.proposed_market_load:
                house.clamps += 1
            house.loads.append(result.market_load)
            house.costs.append(result.cost)
            house.battery_history.append(result.battery_charge)
        records.append(record)
        current_price = record.next_price

    return SimulationResult(houses=houses, records=records, price_history=price_history)


def step_market_hour(
    hour: int,
    price: float,
    house_inputs: list[HouseHourInput],
    config: MarketGameConfig = DEFAULT_CONFIG,
) -> tuple[HourRecord, list[HouseHourResult]]:
    """Validate submitted loads, update batteries, and compute next price."""
    house_results: list[HouseHourResult] = []
    total_market_load = 0.0
    for item in house_inputs:
        clamp = clamp_market_load(
            item.proposed_market_load,
            item.base_demand,
            item.battery.energy,
            config,
        )
        market_load = clamp.market_load
        item.battery.change(market_load - item.base_demand)
        cost = price * market_load
        total_market_load += market_load
        house_results.append(
            HouseHourResult(
                name=item.name,
                proposed_market_load=item.proposed_market_load,
                market_load=market_load,
                base_demand=item.base_demand,
                battery_charge=item.battery.energy,
                cost=cost,
                warning=clamp.warning,
            )
        )

    house_count = len(house_inputs)
    average_market_load = total_market_load / house_count if house_count else 0.0
    next_price = compute_price_from_average_load(average_market_load) if house_count else 0.1
    record = HourRecord(
        hour=hour,
        price=price,
        total_load=total_market_load,
        average_load=average_market_load,
        next_price=next_price,
        loads_by_house={result.name: result.market_load for result in house_results},
        batteries_by_house={result.name: result.battery_charge for result in house_results},
        costs_by_house={result.name: result.cost for result in house_results},
        proposed_loads_by_house={
            result.name: result.proposed_market_load for result in house_results
        },
        warnings_by_house={
            result.name: result.warning for result in house_results if result.warning
        },
    )
    return record, house_results
