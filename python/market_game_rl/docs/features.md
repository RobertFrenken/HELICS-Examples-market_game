# Market Game RL Feature Reference

Notation:

- `t`: current hour.
- `N`: number of houses.
- `L_i,t`: market-facing load submitted by house `i` at hour `t`.
- `L_self,t`: this house's own market-facing load.
- `D_t`: shared base demand at hour `t`.
- `M_t = (1 / N) * sum_i L_i,t`: average market-facing load.
- `p_t`: price observed at hour `t`.

Because the game has a one-hour price lag, `p_t` is caused by `M_{t-1}` for
`t > 0`. The initial `p_0 = 0.5` is exogenous and should not be inverted.

## `invert_price_to_average_load(price)`

Description: estimates the previous-hour average market-facing load from the
current delayed price.

Pricing rule:

```text
p(M) = 0.10                         if M < 3
p(M) = 0.10 + 0.03 * (M - 3)        if 3 <= M < 6
p(M) = 0.19 + 0.10 * (M - 6)        if 6 <= M < 9
p(M) = 0.49 + 0.25 * (M - 9)        if 9 <= M < 13
p(M) = 1.49 + 1.00 * (M - 13)       if M >= 13
```

Inverse estimates:

```text
if p = 0.10:        M is represented as [-10, 3), estimate = 1.5
if 0.10 < p < .19:  M = 3 + (p - 0.10) / 0.03
if 0.19 <= p < .49: M = 6 + (p - 0.19) / 0.10
if 0.49 <= p < 1.49: M = 9 + (p - 0.49) / 0.25
if p >= 1.49:       M = 13 + (p - 1.49)
```

Context: this is the core legal aggregate-observation feature. It reconstructs
a delayed market average, not individual actions.

## `PriceInverse.uncertainty`

Description: width of the feasible inverse interval.

Definition:

```text
uncertainty = inverse_high - inverse_low
```

Context: price `0.10` is lossy because all `M < 3` collapse to the same price.
Policies should be less confident when uncertainty is large.

## `estimate_others_average_load(...)`

Description: removes this house's own previous action from the inferred market
average to estimate the average load of the other houses.

Definition:

```text
others_avg_{t-1} = (N * M_{t-1} - L_self,t-1) / (N - 1)
```

Context: this prevents the policy from mistaking its own market impact for
crowd behavior.

## `crowd_delta`

Description: estimates whether other houses were net charging or discharging
relative to shared base demand.

Definition:

```text
crowd_delta_{t-1} = others_avg_{t-1} - D_{t-1}
```

Interpretation:

```text
crowd_delta > 0  means others were net charging or overbuying
crowd_delta = 0  means others followed base demand on average
crowd_delta < 0  means others were net discharging or underbuying
```

Context: this is a belief feature, not a direct observation.

## `crowd_battery_belief`

Description: rolling belief over the average battery state of other houses.

Definition:

```text
belief_t = clamp(belief_{t-1} + crowd_delta_{t-1}, 0, battery_capacity)
```

Context: useful for estimating whether the crowd can continue suppressing
future demand. It is approximate because a market average hides distribution.

## `distance_to_nearest_pricing_threshold(average_load)`

Description: measures how close inferred average load is to a pricing tier
boundary.

Definition:

```text
tier_distance = min(|M - b| for b in {3, 6, 9, 13})
```

Context: small distance means a modest change in aggregate load may move the
market into a different price-slope region.

## `recent_mean(values, fallback, window)`

Description: rolling average over recent price history.

Definition:

```text
mean_k = (1 / m) * sum(last m values)
m = min(k, len(values))
```

If no values exist, the feature returns `fallback`.

Context: used as a local reference price for deciding whether current price is
cheap or expensive.

## `recent_volatility(values, window)`

Description: rolling population standard deviation over recent price history.

Definition:

```text
sigma_k = sqrt((1 / m) * sum((x_i - mean_k)^2))
m = min(k, len(values))
```

Context: high volatility can justify less aggressive charging/discharging
because the inferred market state is less stable.

## `future_pressure`

Description: short lookahead average of known future base demand.

Definition:

```text
future_pressure_t = (D_{t+1} + D_{t+2} + D_{t+3}) / 3
```

Context: because the full 24-hour demand profile is passed legally into
`compute_demand`, this helps preserve battery before likely high-demand hours.

## `cheap` And `expensive`

Description: compares current price to a recent reference price.

Definitions used by the early heuristics:

```text
cheap      = p_t < cheap_ratio * recent_mean
expensive  = p_t > expensive_ratio * recent_mean
```

`RollingPricePolicy` currently uses `cheap_ratio = 0.94` and
`expensive_ratio = 1.08`. `LegalInferencePolicy` currently uses `0.95` and
`1.05`.

Context: these are intentionally simple baseline features, not final optimized
thresholds.
