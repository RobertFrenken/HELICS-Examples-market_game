"""Unified check runner for market-game RL helpers."""

from __future__ import annotations

import argparse

from .env_check import run_env_smoke_check
from .gym_check import run_gym_smoke_check
from .parity_check import assert_stock_parity


def run_core_checks() -> None:
    assert_stock_parity()
    print("stock parity: ok")
    run_env_smoke_check()
    print("env smoke: ok")
    run_gym_smoke_check()
    print("gym env smoke: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="run market_game_rl checks")
    parser.add_argument(
        "--include-rllib",
        action="store_true",
        help="also run the heavier Ray RLlib PPO smoke check",
    )
    args = parser.parse_args()

    run_core_checks()
    if args.include_rllib:
        from .rllib_check import run_rllib_smoke_check

        run_rllib_smoke_check()
        print("rllib smoke: ok")


if __name__ == "__main__":
    main()
