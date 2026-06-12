"""Unified test runner for upstream-facing market-game helpers."""

from __future__ import annotations

from check_battery_validation import run_check as run_battery_validation_check
from check_house_template import run_check as run_house_template_check
from check_market_maker import run_check as run_market_maker_check


def main() -> None:
    run_battery_validation_check()
    print("battery validation: ok")
    run_house_template_check()
    print("house template: ok")
    run_market_maker_check()
    print("market maker: ok")


if __name__ == "__main__":
    main()
