"""Declarative construction for composed market-game agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .actuators import (
    ActionMapperActuator,
    BatteryDeltaActuator,
    ContinuousNormalizedDeltaActuator,
    DirectLoadActuator,
    IntegerBatteryDeltaActuator,
)
from .actions import DiscreteBatteryPostureActionSpace
from .compose import MarketAgent
from .controllers import FollowDemandController, PriceAwareController
from .observers import InferenceObserver, LocalObserver, PriceHistoryObserver
from .projectors import MarketActionProjector
from .strategies import (
    FollowDemandStrategy,
    FullCycleStrategy,
    PriceAwareStrategy,
    RollingThresholdStrategy,
)


OBSERVER_TYPES = {
    "local": LocalObserver,
    "price_history": PriceHistoryObserver,
    "inference": InferenceObserver,
}

STRATEGY_TYPES = {
    "follow_demand": FollowDemandStrategy,
    "full_cycle": FullCycleStrategy,
    "price_aware": PriceAwareStrategy,
    "rolling_threshold": RollingThresholdStrategy,
}

CONTROLLER_TYPES = {
    "follow_demand": FollowDemandController,
    "price_aware": PriceAwareController,
}

ACTUATOR_TYPES = {
    "direct_load": DirectLoadActuator,
    "battery_delta": BatteryDeltaActuator,
    "battery_posture": lambda: ActionMapperActuator(DiscreteBatteryPostureActionSpace()),
    "integer_battery_delta": lambda **kwargs: ActionMapperActuator(
        IntegerBatteryDeltaActuator(**kwargs)
    ),
    "continuous_normalized_delta": lambda: ActionMapperActuator(
        ContinuousNormalizedDeltaActuator()
    ),
}

ACTION_PROJECTOR_TYPES = {
    "market_action": MarketActionProjector,
}


def build_agent(config: Mapping[str, Any]) -> MarketAgent:
    """Build a composed agent from a plain mapping."""
    agent_config = config.get("agent", config)
    if not isinstance(agent_config, Mapping):
        raise TypeError("agent config must be a mapping")

    name = str(agent_config.get("name", "MarketAgent"))
    if "controller" in agent_config:
        controller = _build_component(agent_config, "controller", CONTROLLER_TYPES)
        action_projector = _build_optional_component(
            agent_config,
            "action_projector",
            ACTION_PROJECTOR_TYPES,
            default=MarketActionProjector(),
        )
        return MarketAgent(
            name=name,
            controller=controller,
            action_projector=action_projector,
        )

    observer = _build_component(agent_config, "observer", OBSERVER_TYPES)
    strategy = _build_component(agent_config, "strategy", STRATEGY_TYPES)
    actuator = _build_component(agent_config, "actuator", ACTUATOR_TYPES)
    return MarketAgent(
        name=name,
        observer=observer,
        strategy=strategy,
        actuator=actuator,
    )


def _build_optional_component(
    agent_config: Mapping[str, Any],
    key: str,
    registry: Mapping[str, Any],
    default: Any,
) -> Any:
    if key not in agent_config:
        return default
    return _build_component(agent_config, key, registry)


def _build_component(
    agent_config: Mapping[str, Any],
    key: str,
    registry: Mapping[str, Any],
) -> Any:
    raw_config = agent_config.get(key)
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"agent.{key} must be a mapping")
    type_name = raw_config.get("type")
    if not isinstance(type_name, str):
        raise ValueError(f"agent.{key}.type must be a string")
    try:
        factory = registry[type_name]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"unknown agent.{key}.type {type_name!r}; choices: {choices}") from exc
    kwargs = {name: value for name, value in raw_config.items() if name != "type"}
    return factory(**kwargs)
