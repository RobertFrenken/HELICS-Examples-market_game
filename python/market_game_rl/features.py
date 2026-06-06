"""Legal feature helpers for market-game policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceInverse:
    estimate: float
    low: float
    high: float

    @property
    def uncertainty(self) -> float:
        return self.high - self.low


def invert_price_to_average_load(price: float) -> PriceInverse:
    """Invert the market price rule into an average-load estimate.

    The flat low-price tier is lossy: price ``0.10`` means average load was
    less than ``3.0``. For that case, return the midpoint estimate ``1.5`` and
    expose the full interval.
    """
    eps = 1e-12
    if price <= 0.10 + eps:
        return PriceInverse(estimate=1.5, low=float("-inf"), high=3.0)
    if price < 0.19 - eps:
        m = 3.0 + (price - 0.10) / 0.03
        return PriceInverse(estimate=m, low=m, high=m)
    if price < 0.49 - eps:
        m = 6.0 + (price - 0.19) / 0.10
        return PriceInverse(estimate=m, low=m, high=m)
    if price < 1.49 - eps:
        m = 9.0 + (price - 0.49) / 0.25
        return PriceInverse(estimate=m, low=m, high=m)
    m = 13.0 + (price - 1.49)
    return PriceInverse(estimate=m, low=m, high=m)


def estimate_others_average_load(
    average_market_load: float,
    own_previous_load: float,
    house_count: int,
) -> float:
    if house_count <= 1:
        return average_market_load
    return (house_count * average_market_load - own_previous_load) / (house_count - 1)

