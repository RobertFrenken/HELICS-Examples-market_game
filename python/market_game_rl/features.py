"""Legal feature helpers for market-game policies."""

from __future__ import annotations

from dataclasses import dataclass


PRICING_THRESHOLDS = (3.0, 6.0, 9.0, 13.0)


@dataclass(frozen=True)
class PriceInverse:
    """Inverse-price feature bundle.

    Let ``p_t`` be the current observed price. Because price is one-hour
    delayed, this estimates ``M_{t-1}``, the previous-hour average market load.

    ``estimate`` is the scalar value used by policies. ``low`` and ``high`` are
    the feasible inverse interval. ``uncertainty = high - low``.
    """

    estimate: float
    low: float
    high: float

    @property
    def uncertainty(self) -> float:
        return self.high - self.low


def invert_price_to_average_load(price: float) -> PriceInverse:
    """Invert the market price rule into an average-load estimate.

    Definition:

    ``M = total_load / N``

    ``p(M) = 0.10`` for ``M < 3``
    ``p(M) = 0.10 + 0.03(M - 3)`` for ``3 <= M < 6``
    ``p(M) = 0.19 + 0.10(M - 6)`` for ``6 <= M < 9``
    ``p(M) = 0.49 + 0.25(M - 9)`` for ``9 <= M < 13``
    ``p(M) = 1.49 + 1.00(M - 13)`` for ``M >= 13``

    The inverse feature estimates ``M`` from an observed delayed price.

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
    """Estimate the previous-hour average load of all other houses.

    Definition:

    ``M = (own_load + sum(other_loads)) / N``

    Solving for the other-house average:

    ``others_avg = (N * M - own_load) / (N - 1)``

    This is legal because it uses only delayed aggregate inference, own action
    history, and known/assumed house count.
    """
    if house_count <= 1:
        return average_market_load
    return (house_count * average_market_load - own_previous_load) / (house_count - 1)


def distance_to_nearest_pricing_threshold(average_load: float) -> float:
    """Measure distance to the nearest pricing tier boundary.

    Definition:

    ``tier_distance = min(|M - b| for b in {3, 6, 9, 13})``

    Smaller values mean the market is close to a price-slope change where small
    changes in aggregate load may have larger future-price consequences.
    """
    return min(abs(average_load - threshold) for threshold in PRICING_THRESHOLDS)


def recent_mean(values: list[float], fallback: float, window: int = 6) -> float:
    """Compute a rolling mean feature.

    Definition for recent values ``x`` and window ``k``:

    ``mean_k = (1 / m) * sum(x_i for i in last m values)``, where
    ``m = min(k, len(values))``.

    If there is no history, return ``fallback``. Policies use this as a local
    price reference without leaking future prices.
    """
    recent = values[-window:]
    if not recent:
        return fallback
    return sum(recent) / len(recent)


def recent_volatility(values: list[float], window: int = 6) -> float:
    """Compute rolling population standard deviation.

    Definition:

    ``sigma_k = sqrt((1 / m) * sum((x_i - mean_k)^2))``, where
    ``m = min(k, len(values))``.

    This summarizes recent price instability. The current implementation uses
    population variance because the feature is descriptive, not an unbiased
    estimator.
    """
    recent = values[-window:]
    if len(recent) < 2:
        return 0.0
    mean = sum(recent) / len(recent)
    variance = sum((value - mean) ** 2 for value in recent) / len(recent)
    return variance**0.5
