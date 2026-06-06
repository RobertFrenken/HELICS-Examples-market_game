"""Example house that cycles the battery without using price.

This strategy is intentionally simple:

1. Charge the battery to full.
2. Discharge the battery to empty.
3. Repeat.

It ignores `price` completely and is useful as a baseline example when you want
to compare a price-aware strategy against a strategy that just moves energy in
and out of the battery.
"""

import argparse

from battery import BATTERY_CAPCITY, BATTERY_MAX_CHARGE, BATTERY_MAX_DISCHARGE
from house_template import House


class FullCycleHouse(House):
    """House that alternates between filling and emptying the battery."""

    def __init__(self, name: str, connection: str = "localhost"):
        super().__init__(name, connection)
        self.charging = True

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        """Return demand that fully charges or discharges regardless of price."""
        del price, price_history

        if self.charging and battery_charge >= BATTERY_CAPCITY:
            self.charging = False
        elif (not self.charging) and battery_charge <= 0.0:
            self.charging = True

        if self.charging:
            charge_amount = min(BATTERY_MAX_CHARGE, BATTERY_CAPCITY - battery_charge)
            return demand[hour] + charge_amount

        discharge_amount = min(BATTERY_MAX_DISCHARGE, battery_charge)
        return demand[hour] - discharge_amount


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="market game house federate commands")
    parser.add_argument("--broker", type=str, default="localhost", help="address of the broker")
    parser.add_argument("--no-plot", action="store_true", default=False, help="skip showing plots at the end")

    args = parser.parse_args()
    house = FullCycleHouse("FullCycleHouse", args.broker)
    house.run()
    if not args.no_plot:
        house.plot_results()
