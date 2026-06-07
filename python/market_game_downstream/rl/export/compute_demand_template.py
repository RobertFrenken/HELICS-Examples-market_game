"""Template for a competition-safe market-game submission.

Copy only the ``compute_demand`` function into the submitted house file unless
the competition instructions explicitly allow helper functions or constants.
"""


def compute_demand(price, hour, battery_charge, demand, price_history):
    """Return the desired market load for one house and one hour."""

    del price_history
    base_demand = demand[hour]
    remaining_capacity = 20.0 - battery_charge

    if price <= 0.12 and remaining_capacity > 0.0:
        return base_demand + min(5.0, remaining_capacity)

    if price >= 0.49 and battery_charge > 0.0:
        return base_demand - min(10.0, battery_charge)

    if hour >= 21 and battery_charge > 0.0:
        return base_demand - min(10.0, battery_charge)

    return base_demand
