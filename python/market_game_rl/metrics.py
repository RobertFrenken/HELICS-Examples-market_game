"""Evaluation metrics for market-game simulation results."""

from __future__ import annotations

from .simulator import SimulationResult


def price_volatility(prices: list[float]) -> float:
    """Sum absolute hour-to-hour price movements."""
    if len(prices) < 2:
        return 0.0
    return sum(abs(prices[index] - prices[index - 1]) for index in range(1, len(prices)))


def result_price_volatility(result: SimulationResult) -> float:
    return price_volatility(result.price_history)
