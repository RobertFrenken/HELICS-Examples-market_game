"""Checks for composed market-game agents."""

from __future__ import annotations

from python.market_game_downstream.core import DEFAULT_CONFIG
from python.market_game_downstream.rl.agents import (
    FollowDemandController,
    MarketAgent,
    PriceAwareController,
    build_agent,
)
from python.market_game_downstream.rl.agents.policies import (
    FollowDemandPolicy,
    PriceAwarePolicy,
)
from python.market_game_downstream.rl.agents.strategies import RollingThresholdStrategy


def run_agents_check() -> None:
    demand = DEFAULT_CONFIG.demand_profile
    price_history = [DEFAULT_CONFIG.initial_price]

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

    follow = build_agent(
        {
            "agent": {
                "name": "composed_follow",
                "observer": {"type": "local"},
                "strategy": {"type": "follow_demand"},
                "actuator": {"type": "direct_load"},
            }
        }
    )
    assert follow.compute_demand(0.10, 0, 5.0, demand, price_history) == (
        FollowDemandPolicy().compute_demand(0.10, 0, 5.0, demand, price_history)
    )

    composed_price_aware = build_agent(
        {
            "agent": {
                "name": "composed_price_aware",
                "observer": {"type": "local"},
                "strategy": {"type": "price_aware"},
                "actuator": {"type": "battery_delta"},
            }
        }
    )
    legacy_price_aware = PriceAwarePolicy()
    for hour in range(DEFAULT_CONFIG.episode_hours):
        for price in (0.10, 0.18, 0.30, 0.55):
            assert composed_price_aware.compute_demand(
                price,
                hour,
                DEFAULT_CONFIG.initial_battery,
                demand,
                price_history,
            ) == legacy_price_aware.compute_demand(
                price,
                hour,
                DEFAULT_CONFIG.initial_battery,
                demand,
                price_history,
            )

    rolling = RollingThresholdStrategy()
    assert rolling.get_params()["reserve"] == 4.0
    rolling.set_params(reserve=2.0)
    assert rolling.get_params()["reserve"] == 2.0


if __name__ == "__main__":
    run_agents_check()
    print("agents: ok")
