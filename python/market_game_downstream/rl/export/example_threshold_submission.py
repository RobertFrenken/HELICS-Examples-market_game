"""Tiny self-contained example submission for local export validation."""


def compute_demand(price, hour, battery_charge, demand, price_history):
    base_demand = demand[hour]
    previous_prices = price_history[:-1]
    reference_price = sum(previous_prices[-6:]) / min(len(previous_prices), 6) if previous_prices else price
    remaining_capacity = 20.0 - battery_charge

    if price < 0.95 * reference_price and remaining_capacity > 0.0:
        return base_demand + min(5.0, remaining_capacity)

    if price > 1.05 * reference_price and battery_charge > 3.0:
        return base_demand - min(10.0, battery_charge - 3.0)

    if hour >= 21 and battery_charge > 0.0:
        return base_demand - min(10.0, battery_charge)

    return base_demand
