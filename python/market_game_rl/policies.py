"""Baseline policies for the pure market-game simulator."""

from __future__ import annotations

from dataclasses import dataclass

from .simulator import BATTERY_CAPACITY, BATTERY_MAX_CHARGE, BATTERY_MAX_DISCHARGE


@dataclass
class FollowDemandPolicy:
    name: str = "FollowDemandHouse"

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        del price, battery_charge, price_history
        return demand[hour]


@dataclass
class FlattenDemandPolicy:
    name: str = "FlattenDemandHouse"

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        del price, price_history
        base_demand = demand[hour]
        target_demand = sum(demand) / len(demand)
        desired_change = target_demand - base_demand

        if desired_change > 0.0:
            charge_amount = min(desired_change, BATTERY_MAX_CHARGE, BATTERY_CAPACITY - battery_charge)
            return base_demand + charge_amount

        discharge_amount = min(abs(desired_change), BATTERY_MAX_DISCHARGE, battery_charge)
        return base_demand - discharge_amount


@dataclass
class FullCyclePolicy:
    name: str = "FullCycleHouse"
    charging: bool = True

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        del price, price_history

        if self.charging and battery_charge >= BATTERY_CAPACITY:
            self.charging = False
        elif (not self.charging) and battery_charge <= 0.0:
            self.charging = True

        if self.charging:
            charge_amount = min(BATTERY_MAX_CHARGE, BATTERY_CAPACITY - battery_charge)
            return demand[hour] + charge_amount

        discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
        return demand[hour] - discharge_amount


@dataclass
class PriceAwarePolicy:
    name: str = "PriceAwareHouse"

    def reserve_target(self, hour: int) -> float:
        if hour < 12:
            return 4.0
        if hour < 18:
            return 8.0
        if hour < 21:
            return 3.0
        return 0.0

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        del price_history

        base_demand = demand[hour]
        remaining_capacity = BATTERY_CAPACITY - battery_charge
        reserve = self.reserve_target(hour)
        available_discharge = max(0.0, battery_charge - reserve)

        if price <= 0.12:
            charge_amount = min(BATTERY_MAX_CHARGE, remaining_capacity)
            return base_demand + charge_amount

        if price <= 0.19 and battery_charge < reserve:
            charge_amount = min(BATTERY_MAX_CHARGE, remaining_capacity, reserve - battery_charge)
            return base_demand + charge_amount

        if price >= 0.49:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
            return base_demand - discharge_amount

        if price >= 0.25 and available_discharge > 0.0:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, available_discharge)
            return base_demand - discharge_amount

        if hour >= 21 and battery_charge > 0.0:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
            return base_demand - discharge_amount

        return base_demand

