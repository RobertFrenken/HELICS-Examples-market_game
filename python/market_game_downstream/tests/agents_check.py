"""Checks for composed market-game agents."""

from __future__ import annotations

from python.market_game_downstream.core import DEFAULT_CONFIG
from python.market_game_downstream.rl.agents import (
    BaseController,
    FlattenDemandController,
    FollowDemandController,
    FullCycleController,
    MarketAgent,
    PriceAwareController,
    RollingThresholdController,
    build_agent,
)
from python.market_game_downstream.rl.agents.policies import (
    FlattenDemandPolicy,
    FullCyclePolicy,
    RollingPricePolicy,
)


def run_agents_check() -> None:
    demand = DEFAULT_CONFIG.demand_profile
    price_history = [DEFAULT_CONFIG.initial_price]

    assert issubclass(FollowDemandController, BaseController)
    assert issubclass(FlattenDemandController, BaseController)
    assert issubclass(FullCycleController, BaseController)
    assert issubclass(PriceAwareController, BaseController)
    assert issubclass(RollingThresholdController, BaseController)

    controller_agent = MarketAgent(
        name="controller_follow",
        controller=FollowDemandController(),
    )
    assert controller_agent.compute_demand(0.10, 2, 5.0, demand, price_history) == demand[2]

    controller_price_aware = MarketAgent(
        name="controller_price_aware",
        controller=PriceAwareController(),
    )
    assert controller_price_aware.compute_demand(0.10, 0, 5.0, demand, price_history) == (
        demand[0] + DEFAULT_CONFIG.max_charge
    )

    controller_flatten = MarketAgent(
        name="controller_flatten",
        controller=FlattenDemandController(),
    )
    assert controller_flatten.compute_demand(0.10, 0, 5.0, demand, price_history) == (
        FlattenDemandPolicy().compute_demand(0.10, 0, 5.0, demand, price_history)
    )

    controller_full_cycle = MarketAgent(
        name="controller_full_cycle",
        controller=FullCycleController(),
    )
    legacy_full_cycle = FullCyclePolicy()
    for battery_charge in (0.0, DEFAULT_CONFIG.battery_capacity, 5.0):
        assert controller_full_cycle.compute_demand(
            0.10,
            0,
            battery_charge,
            demand,
            price_history,
        ) == legacy_full_cycle.compute_demand(
            0.10,
            0,
            battery_charge,
            demand,
            price_history,
        )

    declarative_price_aware = build_agent(
        {
            "agent": {
                "name": "declarative_price_aware",
                "controller": {"type": "price_aware"},
                "action_projector": {"type": "market_action"},
            }
        }
    )
    assert declarative_price_aware.compute_demand(0.55, 12, 5.0, demand, price_history) == (
        demand[12] - 5.0
    )

    declarative_rolling = build_agent(
        {
            "agent": {
                "name": "declarative_rolling",
                "controller": {"type": "rolling_threshold", "reserve": 2.0},
            }
        }
    )
    assert declarative_rolling.compute_demand(
        0.40,
        12,
        5.0,
        demand,
        [0.25, 0.25, 0.25],
    ) == (demand[12] - 3.0)

    rolling = RollingThresholdController()
    assert rolling.get_params()["reserve"] == 4.0
    rolling.set_params(reserve=2.0)
    assert rolling.get_params()["reserve"] == 2.0
    assert RollingPricePolicy(reserve=2.0).compute_demand(
        0.40,
        12,
        5.0,
        demand,
        [0.25, 0.25, 0.25],
    ) == (demand[12] - 3.0)


if __name__ == "__main__":
    run_agents_check()
    print("agents: ok")
