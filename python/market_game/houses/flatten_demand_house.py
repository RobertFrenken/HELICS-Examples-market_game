"""Example house that smooths demand without using price.

This strategy tries to make the market-facing demand as flat as possible over
the full day. It computes a constant target equal to the average daily demand,
then uses the battery to move each hour's demand toward that target.

Because the battery starts empty and has charge/discharge limits, it cannot
always hit the target exactly. It simply moves as close as possible each hour.
"""

import argparse

from battery import BATTERY_CAPCITY, BATTERY_MAX_CHARGE, BATTERY_MAX_DISCHARGE
from house_template import House


class FlattenDemandHouse(House):
    """House that uses the battery to smooth demand toward a flat daily level."""

    def __init__(self, name: str, connection: str = "localhost"):
        super().__init__(name, connection)
        self.target_demand = sum(self.demand) / len(self.demand)

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        """Return demand as close as possible to a flat daily target."""
        del price, price_history

        base_demand = demand[hour]
        desired_change = self.target_demand - base_demand

        if desired_change > 0.0:
            charge_amount = min(
                desired_change,
                BATTERY_MAX_CHARGE,
                BATTERY_CAPCITY - battery_charge,
            )
            return base_demand + charge_amount

        discharge_amount = min(
            abs(desired_change),
            BATTERY_MAX_DISCHARGE,
            battery_charge,
        )
        return base_demand - discharge_amount


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="market game house federate commands")
    parser.add_argument("--broker", type=str, default="localhost", help="address of the broker")
    parser.add_argument("--no-plot", action="store_true", default=False, help="skip showing plots at the end")

    args = parser.parse_args()
    house = FlattenDemandHouse("FlattenDemandHouse", args.broker)
    house.run()
    if not args.no_plot:
        house.plot_results()
