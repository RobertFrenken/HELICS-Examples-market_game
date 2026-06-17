"""Export a tiny RLlib PPO checkpoint as a standalone submission file."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import ray
from ray.tune.registry import register_env

from python.market_game_downstream.rl.agents.observations import ObservationMode
from python.market_game_downstream.rl.export_profiles import (
    EXPORTABLE_PPO_PROFILE,
    ExportProfile,
)
from python.market_game_downstream.rl.training.train_rllib import (
    ENV_NAME,
    build_ppo_config,
    make_env,
)
from .validators import validate_submission_file


def export_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    scenario: str | None = None,
    scenario_seed: int | None = None,
    observation_mode: ObservationMode | str = ObservationMode.PRICE_HISTORY,
    hidden_size: int = 8,
    profile: ExportProfile | None = None,
) -> Path:
    """Write a standalone ``compute_demand`` submission from a small PPO actor."""

    profile = profile or ExportProfile(
        observation_mode=ObservationMode(observation_mode),
        fcnet_hiddens=(hidden_size,),
    )
    profile.validate()
    checkpoint_path = Path(checkpoint).resolve()
    output_path = Path(output)
    state = _load_actor_state(
        checkpoint_path,
        scenario=scenario,
        scenario_seed=scenario_seed,
        observation_mode=profile.observation_mode,
        hidden_size=profile.hidden_size,
    )
    source = render_submission_source(state)
    source_size = len(source.encode("utf-8"))
    if source_size > profile.max_source_bytes:
        raise ValueError(
            f"exported source is {source_size} bytes; profile limit is "
            f"{profile.max_source_bytes} bytes"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    validate_submission_file(output_path)
    return output_path


def _load_actor_state(
    checkpoint_path: Path,
    *,
    scenario: str | None,
    scenario_seed: int | None,
    observation_mode: ObservationMode | str,
    hidden_size: int,
) -> dict[str, Any]:
    register_env(ENV_NAME, make_env)
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level=logging.ERROR,
        num_cpus=1,
    )
    algorithm = (
        build_ppo_config(
            observation_mode=observation_mode,
            scenario=scenario,
            scenario_seed=scenario_seed,
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [hidden_size],
                "fcnet_activation": "tanh",
            }
        )
        .build_algo()
    )
    try:
        algorithm.restore(checkpoint_path.as_posix())
        module = algorithm.get_module()
        state = module.state_dict()
        return _extract_single_hidden_tanh_actor(state, hidden_size)
    finally:
        algorithm.stop()
        ray.shutdown()


def _extract_single_hidden_tanh_actor(state: dict[str, Any], hidden_size: int) -> dict[str, Any]:
    required = {
        "w1": "encoder.actor_encoder.net.mlp.0.weight",
        "b1": "encoder.actor_encoder.net.mlp.0.bias",
        "w2": "pi.net.mlp.0.weight",
        "b2": "pi.net.mlp.0.bias",
    }
    missing = [name for name in required.values() if name not in state]
    if missing:
        raise ValueError(f"checkpoint does not match the supported tiny MLP layout: {missing}")

    w1 = _tensor_to_nested_floats(state[required["w1"]])
    b1 = _tensor_to_nested_floats(state[required["b1"]])
    w2 = _tensor_to_nested_floats(state[required["w2"]])
    b2 = _tensor_to_nested_floats(state[required["b2"]])
    if len(w1) != hidden_size or len(b1) != hidden_size:
        raise ValueError("hidden-size argument does not match checkpoint actor")
    if len(w1[0]) != 10:
        raise ValueError("only price_history observations with 10 features are supported")
    if len(w2) != 3 or any(len(row) != hidden_size for row in w2):
        raise ValueError("only 3-action discrete battery posture actors are supported")
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _tensor_to_nested_floats(value: Any) -> Any:
    return value.detach().cpu().numpy().round(8).tolist()


def render_submission_source(state: dict[str, Any]) -> str:
    """Render a validator-safe standalone Python submission."""

    return f'''"""Standalone tiny PPO policy exported from RLlib."""

import math

W1 = {_format_literal(state["w1"])}
B1 = {_format_literal(state["b1"])}
W2 = {_format_literal(state["w2"])}
B2 = {_format_literal(state["b2"])}


def _dot(row, values):
    total = 0.0
    for index, weight in enumerate(row):
        total += weight * values[index]
    return total


def _features(price, hour, battery_charge, demand, price_history):
    angle = 2.0 * math.pi * hour / 24.0
    base_demand = demand[hour]
    previous_prices = price_history[:-1]
    lag_1 = previous_prices[-1] if len(previous_prices) >= 1 else price
    lag_2 = previous_prices[-2] if len(previous_prices) >= 2 else price
    recent = previous_prices[-6:]
    price_mean = sum(recent) / len(recent) if recent else price
    price_trend = previous_prices[-1] - previous_prices[-2] if len(previous_prices) >= 2 else 0.0
    if len(recent) <= 1:
        price_volatility = 0.0
    else:
        recent_mean = sum(recent) / len(recent)
        variance = sum((value - recent_mean) ** 2 for value in recent) / len(recent)
        price_volatility = math.sqrt(variance)
    return [
        math.sin(angle),
        math.cos(angle),
        price,
        battery_charge,
        base_demand,
        lag_1,
        lag_2,
        price_mean,
        price_trend,
        price_volatility,
    ]


def _action(features):
    hidden = []
    for row, bias in zip(W1, B1):
        hidden.append(math.tanh(_dot(row, features) + bias))
    logits = []
    for row, bias in zip(W2, B2):
        logits.append(_dot(row, hidden) + bias)
    best = 0
    if logits[1] > logits[best]:
        best = 1
    if logits[2] > logits[best]:
        best = 2
    return best


def compute_demand(price, hour, battery_charge, demand, price_history):
    base_demand = demand[hour]
    action = _action(_features(price, hour, battery_charge, demand, price_history))
    if action == 0:
        return base_demand - min(10.0, battery_charge)
    if action == 2:
        return base_demand + min(5.0, 20.0 - battery_charge)
    return base_demand
'''


def _format_literal(value: Any) -> str:
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="export a tiny one-hidden-layer RLlib PPO checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--scenario-seed", type=int)
    parser.add_argument(
        "--observation-mode",
        choices=[mode.value for mode in ObservationMode],
        default=ObservationMode.PRICE_HISTORY.value,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=EXPORTABLE_PPO_PROFILE.hidden_size,
        help="single hidden-layer size supported by the exporter",
    )
    args = parser.parse_args()
    output = export_checkpoint(
        args.checkpoint,
        args.output,
        scenario=args.scenario,
        scenario_seed=args.scenario_seed,
        observation_mode=args.observation_mode,
        hidden_size=args.hidden_size,
    )
    print(f"exported={output}")


if __name__ == "__main__":
    main()
