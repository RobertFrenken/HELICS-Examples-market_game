"""Example player house for the HELICS market game.

If you are using this file as a starting point, there are usually only two
things to change:

1. Rename the class to something that matches your house.
2. Replace `compute_demand()` with your own strategy.

You normally do not need to edit the `__main__` block except to give your house
an identifying name.
"""

from house_template import House

import argparse

class TestHouse(House):
    """Simple example strategy.

    This house charges a little when the battery is empty and discharges a
    little when it has stored energy.
    """

    def __init__(self, name: str, connection: str = "localhost"):
        super().__init__(name, connection)

    def compute_demand(self, price: float, hour: int, battery_charge: float, demand: list[float], price_history: list[float]) -> float:
        """Return this hour's market demand.

        Edit this method to create your own strategy.

        Quick reminder:
        - return `demand[hour]` to leave the battery unchanged
        - return more than `demand[hour]` to charge the battery
        - return less than `demand[hour]` to discharge the battery
        """
        # Example strategy: charge by 2 kWh until the battery has energy,
        # then discharge by 2 kWh.
        if battery_charge == 0:
            return demand[hour] + 2
        else:
            return demand[hour] - 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="market game house federate commands")
    parser.add_argument("--broker", type=str, default="localhost", help="address of the broker")
    parser.add_argument("--no-plot", action="store_true", default=False, help="skip showing plots at the end")

    args = parser.parse_args()
    # Change "TestHouse" to the unique player name you want shown in the game.
    house=TestHouse("TestHouse",args.broker)
    house.run()
    if not args.no_plot:
        house.plot_results()
    
