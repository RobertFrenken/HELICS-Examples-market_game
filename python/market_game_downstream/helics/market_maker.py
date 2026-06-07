from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from typing import Any

from battery import Battery
from python.market_game_downstream.core.config import DEFAULT_CONFIG, demand_profile
from python.market_game_downstream.core.rules import compute_price_from_total_load
from python.market_game_downstream.core.simulator import HouseHourInput, HouseHourResult, step_market_hour


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
    """Compatibility wrapper for the shared price rule."""
    return compute_price_from_total_load(total, feds)


def update_demand(type: str, fed: SubFed) -> None:
    """Compatibility wrapper for assigning a named demand profile."""
    fed.demand = demand_profile(type)


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


def read_hour_inputs(h, feds: list[SubFed], hour: int) -> list[HouseHourInput]:
    return [
        HouseHourInput(
            name=fed.name,
            proposed_market_load=h.helicsInputGetDouble(fed.input),
            base_demand=fed.demand[hour],
            battery=fed.battery,
        )
        for fed in feds
    ]


def apply_hour_results(feds: list[SubFed], results: list[HouseHourResult], hour: int) -> None:
    for fed, result in zip(feds, results):
        if result.warning:
            print(
                f"invalid demand received for fed {fed.name}={result.proposed_market_load} "
                f"vs {result.market_load} warning={result.warning}, recalculating with new value "
                f"and assessing penalty={result.penalty_cost}"
            )
        fed.consume.append(result.market_load)
        fed.hourCost.append(result.cost)
        print(
            f"hr {hour}: federate {fed.name} using {result.market_load} scheduled "
            f"{result.base_demand} battery at {result.battery_charge} cost={result.cost}"
        )


def print_winner(feds: list[SubFed]) -> None:
    low_fed_cost: float = 1000000000000.0
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

    price_publication = market_maker.register_global_publication(
        "price",
        h.HELICS_DATA_TYPE_DOUBLE,
        "$/kWh",
    )
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
    current_price = DEFAULT_CONFIG.initial_price
    price_publication.publish(current_price)
    h.helicsFederateEnterExecutingMode(market_maker)

    current_time = 0
    prices = []
    loads = []
    while current_time < DEFAULT_CONFIG.episode_hours:
        hour = int(current_time)
        record, results = step_market_hour(
            hour=hour,
            price=current_price,
            house_inputs=read_hour_inputs(h, feds, hour),
        )
        apply_hour_results(feds, results, hour)
        loads.append(record.total_load)
        print(f"hr {hour}: total load {record.total_load} new  price = {current_price} ")
        current_price = record.next_price
        price_publication.publish(current_price)
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
