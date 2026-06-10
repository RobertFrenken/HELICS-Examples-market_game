import argparse
from dataclasses import dataclass, field
import json
import random
from typing import Any

from battery import Battery, check_valid, ensure_valid


@dataclass
class SubFed:
    battery: Battery = field(default_factory=Battery)
    input: Any = None
    totalCost: float = 0.0
    hourCost: list[float] = field(default_factory=list)
    demand: list[float] = field(default_factory=lambda: [5] * 24)
    consume: list[float] = field(default_factory=list)
    name: str = ""


def compute_new_price(total: float, feds: int) -> float:
    if feds == 0:
        return 0.1

    average_load = total / feds
    if average_load < 3.0:
        return 0.1
    if average_load < 6.0:
        return 0.1 + 0.03 * (average_load - 3.0)
    if average_load < 9.0:
        return 0.19 + 0.1 * (average_load - 6.0)
    if average_load < 13.0:
        return 0.49 + 0.25 * (average_load - 9.0)
    return 1.49 + 1.0 * (average_load - 13.0)


def update_demand(profile_type: str, fed: SubFed) -> None:
    if profile_type == "random":
        elements = [random.random() for _ in range(24)]
        multiplier = 120.0 / sum(elements)
        fed.demand = [value * multiplier for value in elements]
    elif profile_type == "spike":
        fed.demand = [4] * 24
        fed.demand[random.randint(0, 23)] = 28
    elif profile_type == "dspike":
        fed.demand = [3] * 24
        fed.demand[random.randint(0, 23)] += 24
        fed.demand[random.randint(0, 23)] += 24
    elif profile_type == "profile1":
        fed.demand = [
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
    elif profile_type == "profile_solar":
        profile = [
            2,
            2,
            2,
            2,
            3,
            4,
            5,
            2,
            -4,
            -6,
            -7,
            -8,
            -7,
            -6,
            -3,
            1,
            4,
            8,
            10,
            11,
            10,
            7,
            5,
            3,
        ]
        fed.demand = [value * 3.0 for value in profile]
    else:
        fed.demand = [5] * 24


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="market maker federate commands")
    parser.add_argument("--auto", action="store_true", default=False, help="run in auto mode")
    parser.add_argument("--autobroker", action="store_true", default=False, help="enable local broker")
    parser.add_argument("--broker", type=str, default="localhost", help="address of the broker")
    parser.add_argument("--no-plot", action="store_true", default=False, help="skip showing plots at the end")
    parser.add_argument(
        "--profile",
        type=str,
        default="profile1",
        help="type of load profile to use: flat, spike, dspike, random, profile1, profile_solar",
    )
    return parser


def query_federates(h, fed_query, market_maker, broker, use_broker: bool):
    if use_broker:
        return h.helicsQueryBrokerExecute(fed_query, broker)
    return h.helicsQueryExecute(fed_query, market_maker)


def wait_for_manual_start(h, fed_query, market_maker, broker, use_broker: bool) -> bool:
    while True:
        results = query_federates(h, fed_query, market_maker, broker, use_broker)
        for index, fed in enumerate(results):
            print(f"fed {index}:{fed}")

        response = input("enter i to start initialization: ")
        if response and response[0] == "i":
            return True
        if response and response[0] == "q":
            return False


def register_house_feds(h, market_maker, federate_names: list[str], profile: str) -> list[SubFed]:
    feds: list[SubFed] = []
    for fed_name in federate_names:
        print(f"fed {len(feds)}:{fed_name}")
        subfed = SubFed(name=fed_name)
        if profile != "flat":
            update_demand(profile, subfed)
        subfed.input = h.helicsFederateRegisterSubscription(
            market_maker,
            subfed.name + "/demand",
            "kWh",
        )
        h.helicsFederateSendCommand(market_maker, subfed.name, json.dumps({"demand": subfed.demand}))
        feds.append(subfed)
    return feds


def run_market_hour(h, feds: list[SubFed], hour: int, current_price: float) -> float:
    total_load = 0.0
    for fed in feds:
        penalty_cost = 0.0
        load = h.helicsInputGetDouble(fed.input)
        warning = check_valid(load, fed.demand[hour], fed.battery)
        if warning:
            valid_load = ensure_valid(load, fed.demand[hour], fed.battery)
            penalty_cost = 20 * abs(load - valid_load)
            print(
                f"invalid demand received for fed {fed.name}={load} vs {valid_load} "
                f"warning={warning}, recalculating with new value and "
                f"assessing penalty={penalty_cost}"
            )
            load = valid_load

        fed.battery.change(load - fed.demand[hour])
        fed.consume.append(load)
        hour_cost = load * current_price + penalty_cost
        fed.hourCost.append(hour_cost)
        print(
            f"hr {hour}: federate {fed.name} using {load} scheduled {fed.demand[hour]} "
            f"battery at {fed.battery.energy} cost={hour_cost}"
        )
        total_load += load
    return total_load


def print_winner(feds: list[SubFed]) -> None:
    low_fed_cost = float("inf")
    low_fed = None
    for fed in feds:
        fed.totalCost = float(sum(fed.hourCost))
        if fed.totalCost < low_fed_cost:
            low_fed = fed
            low_fed_cost = fed.totalCost
        print(f"fed {fed.name}:total Cost=${fed.totalCost} total consumption={sum(fed.consume)}")

    if low_fed is not None:
        print(f"the winner is Fed {low_fed.name} total cost=${low_fed_cost}")


def plot_market_summary(loads: list[float], prices: list[float]) -> None:
    import matplotlib.pyplot as plt

    time = list(range(24))
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(time, loads, color="g", label="load")
    axis[0].set_title("Demand profile")
    axis[1].plot(time, prices)
    axis[1].set_title("prices")
    plt.show()


def main() -> None:
    import helics as h

    args = build_parser().parse_args()

    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreType(fedinfo, h.HELICS_CORE_TYPE_ZMQ_SS)
    h.helicsFederateInfoSetBroker(fedinfo, args.broker)
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_period, 1.0)
    h.helicsFederateInfoSetFlagOption(fedinfo, h.HELICS_FLAG_WAIT_FOR_CURRENT_TIME_UPDATE, True)

    federate_name = "market_maker_fed"
    broker = None
    if args.autobroker:
        broker = h.helicsCreateBroker("zmqss", "market_maker", "--ipv4 -f1")
        print(f"broker created: address= {h.helicsBrokerGetAddress(broker)}")

    market_maker = h.helicsCreateCombinationFederate(federate_name, fedinfo)
    price = market_maker.register_global_publication("price", h.HELICS_DATA_TYPE_DOUBLE, "$/kWh")
    fed_query = h.helicsCreateQuery("federation", "federates")

    if not args.auto and not wait_for_manual_start(
        h,
        fed_query,
        market_maker,
        broker,
        args.autobroker,
    ):
        h.helicsFederateDisconnect(market_maker)
        if args.autobroker:
            h.helicsBrokerDisconnect(broker)
        return

    h.helicsFederateEnterInitializingModeIterative(market_maker)
    federation_members = query_federates(h, fed_query, market_maker, broker, args.autobroker)
    house_names = [fed for fed in federation_members if fed != federate_name]
    feds = register_house_feds(h, market_maker, house_names, args.profile)

    h.helicsFederateEnterInitializingMode(market_maker)
    current_price = 0.5
    price.publish(current_price)
    h.helicsFederateEnterExecutingMode(market_maker)

    current_time = 0
    prices = []
    loads = []
    while current_time < 24:
        hour = int(current_time)
        total_load = run_market_hour(h, feds, hour, current_price)
        loads.append(total_load)
        print(f"hr {hour}: total load {total_load} new  price = {current_price} ")
        current_price = compute_new_price(total_load, len(feds))
        price.publish(current_price)
        prices.append(current_price)
        current_time = market_maker.request_next_step()

    h.helicsFederateDisconnect(market_maker)
    print_winner(feds)
    if args.autobroker:
        h.helicsBrokerDisconnect(broker)
    if not args.no_plot:
        plot_market_summary(loads, prices)


if __name__ == "__main__":
    main()
