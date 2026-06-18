"""Generic HELICS house wrapper for local policy experiments.

This keeps the canonical HELICS runtime in ``python/market_game`` while letting
downstream training and export tools launch policy objects or standalone
``compute_demand(...)`` files as ordinary market-game houses.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable

from house_template import House


ComputeDemand = Callable[[float, int, float, list[float], list[float]], float]


class PolicyHouse(House):
    """HELICS house backed by any object exposing ``compute_demand(...)``."""

    def __init__(
        self,
        name: str,
        compute_demand: ComputeDemand,
        connection: str = "localhost",
    ):
        self._compute_demand = compute_demand
        super().__init__(name, connection)

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        return float(
            self._compute_demand(
                price,
                hour,
                battery_charge,
                demand,
                price_history,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run a policy-backed market-game house")
    parser.add_argument("--broker", type=str, default="localhost", help="address of the broker")
    parser.add_argument("--no-plot", action="store_true", default=False, help="skip showing plots at the end")
    parser.add_argument("--name", required=True, help="HELICS federate and output name")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--policy-type",
        help="downstream policy type name, such as PriceAwarePolicy",
    )
    source.add_argument(
        "--submission",
        help="standalone Python file defining compute_demand(...)",
    )
    parser.add_argument(
        "--policy-kwargs",
        default="{}",
        help="JSON keyword arguments for --policy-type",
    )
    return parser


def downstream_policy_compute_demand(policy_type: str, kwargs_json: str) -> ComputeDemand:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from python.market_game_downstream.rl.agents.policies import POLICY_TYPES
    from python.market_game_downstream.rl.agents.callables import policy_to_compute_demand

    try:
        policy_class = POLICY_TYPES[policy_type]
    except KeyError as exc:
        choices = ", ".join(sorted(POLICY_TYPES))
        raise ValueError(f"unknown policy type {policy_type!r}; choices: {choices}") from exc

    kwargs = json.loads(kwargs_json)
    if not isinstance(kwargs, dict):
        raise ValueError("--policy-kwargs must decode to a JSON object")
    policy = policy_class(**kwargs)
    return policy_to_compute_demand(policy)


def submission_compute_demand(path: str) -> ComputeDemand:
    module = _load_module(Path(path))
    compute_demand = getattr(module, "compute_demand", None)
    if not callable(compute_demand):
        raise ValueError(f"submission {path!r} must define compute_demand(...)")
    return compute_demand


def _load_module(path: Path) -> ModuleType:
    resolved = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("market_game_submission", resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load submission {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = build_parser().parse_args()
    if args.policy_type:
        compute_demand = downstream_policy_compute_demand(
            args.policy_type,
            args.policy_kwargs,
        )
    else:
        compute_demand = submission_compute_demand(args.submission)
    house = PolicyHouse(args.name, compute_demand, args.broker)
    house.run()
    if not args.no_plot:
        house.plot_results()


if __name__ == "__main__":
    main()
