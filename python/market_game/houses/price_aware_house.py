"""Example house that responds to price in a simple, sensible way.

This strategy uses the published market tiers directly:

- charge aggressively when price is cheap
- build some reserve when price is still fairly low
- discharge when price is expensive
- empty the battery near the end of the day instead of leaving energy unused

It is not meant to be optimal. It is meant to be easy to read and to provide a
useful starting point for players who want a price-aware strategy.
"""

import argparse

from battery import BATTERY_CAPCITY, BATTERY_MAX_CHARGE, BATTERY_MAX_DISCHARGE
from house_template import House


class PriceAwareHouse(House):
    """House that charges low and discharges high using simple thresholds."""

    def __init__(self, name: str, connection: str = "localhost"):
        super().__init__(name, connection)

    def reserve_target(self, hour: int) -> float:
        """Keep some stored energy available for later expensive hours."""
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
        """Return demand using simple price thresholds and battery reserve."""
        del price_history

        base_demand = demand[hour]
        remaining_capacity = BATTERY_CAPCITY - battery_charge
        reserve = self.reserve_target(hour)
        available_discharge = max(0.0, battery_charge - reserve)

        # Very cheap power: buy extra and store as much as allowed.
        if price <= 0.12:
            charge_amount = min(BATTERY_MAX_CHARGE, remaining_capacity)
            return base_demand + charge_amount

        # Moderately cheap power: keep building a reserve for later.
        if price <= 0.19 and battery_charge < reserve:
            charge_amount = min(BATTERY_MAX_CHARGE, remaining_capacity, reserve - battery_charge)
            return base_demand + charge_amount

        # Very expensive power: discharge hard to avoid buying from the market.
        if price >= 0.49:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
            return base_demand - discharge_amount

        # Expensive power: discharge, but try to keep some reserve unless late.
        if price >= 0.25 and available_discharge > 0.0:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, available_discharge)
            return base_demand - discharge_amount

        # End of day: use remaining stored energy rather than leaving it unused.
        if hour >= 21 and battery_charge > 0.0:
            discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
            return base_demand - discharge_amount

        return base_demand


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="market game house federate commands")
    parser.add_argument("--broker", type=str, default="localhost", help="address of the broker")
    parser.add_argument("--no-plot", action="store_true", default=False, help="skip showing plots at the end")

    args = parser.parse_args()
    house = PriceAwareHouse("PriceAwareHouse", args.broker)
    house.run()
    if not args.no_plot:
        house.plot_results()
