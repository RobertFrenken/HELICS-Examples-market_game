"""Declarative construction for composed market-game agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .compose import MarketAgent
from .controllers import (
    BatteryDeltaDecoder,
    BatteryPostureIndexDecoder,
    FlattenDemandController,
    FollowDemandController,
    FullCycleController,
    InvalidDemandController,
    LegalInferenceController,
    NoisyThresholdController,
    NormalizedBatteryDeltaDecoder,
    OscillatingController,
    PriceAwareController,
    RollingThresholdController,
    TargetLoadDecoder,
    TinyTanhController,
    VolatilitySeekingController,
)
from ..observations import (
    InferenceFeatureExtractor,
    LocalFeatureExtractor,
    PriceHistoryFeatureExtractor,
)
from .state import DictAgentState, InferenceBeliefState, NoAgentState

CONTROLLER_TYPES = {
    "flatten_demand": FlattenDemandController,
    "follow_demand": FollowDemandController,
    "full_cycle": FullCycleController,
    "invalid_demand": InvalidDemandController,
    "legal_inference": LegalInferenceController,
    "noisy_threshold": NoisyThresholdController,
    "oscillating": OscillatingController,
    "price_aware": PriceAwareController,
    "rolling_threshold": RollingThresholdController,
    "tiny_tanh": TinyTanhController,
    "volatility_seeking": VolatilitySeekingController,
}

FEATURE_EXTRACTOR_TYPES = {
    "local": LocalFeatureExtractor,
    "price_history": PriceHistoryFeatureExtractor,
    "inference": InferenceFeatureExtractor,
}

ACTION_DECODER_TYPES = {
    "battery_delta": BatteryDeltaDecoder,
    "battery_posture_index": BatteryPostureIndexDecoder,
    "normalized_battery_delta": NormalizedBatteryDeltaDecoder,
    "target_load": TargetLoadDecoder,
}

STATE_TYPES = {
    "none": NoAgentState,
    "dict": DictAgentState,
    "inference_belief": InferenceBeliefState,
}


def build_agent(config: Mapping[str, Any]) -> MarketAgent:
    """Build a composed agent from a plain mapping."""
    agent_config = config.get("agent", config)
    if not isinstance(agent_config, Mapping):
        raise TypeError("agent config must be a mapping")

    name = str(agent_config.get("name", "MarketAgent"))
    controller = _build_controller(agent_config)
    state = _build_optional_component(
        agent_config,
        "state",
        STATE_TYPES,
        default=NoAgentState(),
    )
    return MarketAgent(
        name=name,
        controller=controller,
        state=state,
    )


def _build_controller(agent_config: Mapping[str, Any]) -> Any:
    raw_config = agent_config.get("controller")
    if not isinstance(raw_config, Mapping):
        raise ValueError("agent.controller must be a mapping")
    type_name = raw_config.get("type")
    if not isinstance(type_name, str):
        raise ValueError("agent.controller.type must be a string")
    try:
        factory = CONTROLLER_TYPES[type_name]
    except KeyError as exc:
        choices = ", ".join(sorted(CONTROLLER_TYPES))
        raise ValueError(
            f"unknown agent.controller.type {type_name!r}; choices: {choices}"
        ) from exc
    kwargs = {name: value for name, value in raw_config.items() if name != "type"}
    if "feature_extractor" in kwargs:
        kwargs["feature_extractor"] = _build_nested_component(
            kwargs["feature_extractor"],
            "agent.controller.feature_extractor",
            FEATURE_EXTRACTOR_TYPES,
        )
    if "action_decoder" in kwargs:
        kwargs["action_decoder"] = _build_nested_component(
            kwargs["action_decoder"],
            "agent.controller.action_decoder",
            ACTION_DECODER_TYPES,
        )
    return factory(**kwargs)


def _build_optional_component(
    agent_config: Mapping[str, Any],
    key: str,
    registry: Mapping[str, Any],
    default: Any,
) -> Any:
    if key not in agent_config:
        return default
    return _build_component(agent_config, key, registry)


def _build_nested_component(
    raw_config: Any,
    label: str,
    registry: Mapping[str, Any],
) -> Any:
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"{label} must be a mapping")
    type_name = raw_config.get("type")
    if not isinstance(type_name, str):
        raise ValueError(f"{label}.type must be a string")
    try:
        factory = registry[type_name]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"unknown {label}.type {type_name!r}; choices: {choices}") from exc
    kwargs = {name: value for name, value in raw_config.items() if name != "type"}
    return factory(**kwargs)


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
