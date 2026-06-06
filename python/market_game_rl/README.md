# Market Game RL Helpers

This directory contains a pure-Python simulator and training/evaluation helpers
for the HELICS market game.

For module boundaries and dependency flow, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

It intentionally does not import from `python/market_game`.

Reasons:

- `market_maker.py` executes the HELICS market loop at import time.
- `house_template.py` imports HELICS and matplotlib.
- RL needs a fast single-process inner loop.
- The pure simulator should be usable without broker/process orchestration.

The simulator mirrors the inspected game mechanics:

- initial price `0.5`;
- 24-hour episode;
- current price included in `price_history`;
- one-hour price lag;
- pricing from average market-facing load;
- battery capacity `20`, max charge `5`, max discharge `10`;
- market-maker effective clamping;
- negative market-facing load allowed when battery discharge supports it.

## Package Layout

```text
python/market_game_rl/
  config.py        episode config and default constants
  rules.py         pricing, clamping, and discrete action conversion
  simulator.py     pure state transition loop
  observations.py  legal observation builders
  env.py           dependency-free Gymnasium-style environment
  gym_env.py       optional Gymnasium adapter
  train_rllib.py   minimal Ray RLlib PPO trainer
  ARCHITECTURE.md  package boundaries and dependency flow
  policies.py      baseline and heuristic policies
  features.py      legal feature helpers and formulas
  metrics.py       evaluation metrics
  evaluate.py      CLI scenario runner
  parity_check.py  executable stock parity assertions
  env_check.py     executable environment smoke checks
  gym_check.py     executable Gymnasium adapter checks
  rllib_check.py   executable RLlib PPO smoke check
```

## Parity Check

From the repo root:

```bash
python3 -m python.market_game_rl.evaluate --stock
```

Expected stock `profile1` output:

```text
agent,total_load,total_cost,final_battery
FlattenDemandHouse,127.0000000000,35.4466666667,7.0000000000
FullCycleHouse,120.0000000000,47.7466666667,0.0000000000
PriceAwareHouse,125.0000000000,21.9533333333,5.0000000000
```

These values match a live HELICS run of the included example houses.

## Baseline Evaluation

From the repo root:

```bash
python3 -m python.market_game_rl.evaluate
```

This prints CSV rows for the stock example, all-follow-demand baseline, and
early heuristic mixes. The `boundary_warnings` column mirrors the original
inclusive boundary warnings. The `clamps` column counts effective value changes
and is the better constraint-quality metric.

The executable parity assertion is:

```bash
python3 -m python.market_game_rl.parity_check
```

The dependency-free environment smoke check is:

```bash
python3 -m python.market_game_rl.env_check
```

The optional Gymnasium adapter smoke check is:

```bash
python3 -m python.market_game_rl.gym_check
```

The optional Ray RLlib smoke check is:

```bash
python3 -m python.market_game_rl.rllib_check
```

Run a short PPO training job:

```bash
python3 -m python.market_game_rl.train_rllib --iterations 1 --observation-mode price_history
```

Ray, RLlib, and Torch are offline-training dependencies only. Deployment should
still use a self-contained `compute_demand(...)` policy with no Ray/Torch
runtime dependency.

## Environment Interface

`MarketGameEnv` is Gymnasium-like but does not require Gymnasium:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

Actions use `BatteryAction`:

```text
-1  discharge
 0  neutral
 1  charge
```

The current terminal observation is built from the last valid episode hour.
Hidden aggregate market values are available only in `info["diagnostics"]`,
not in the observation vector.

`GymMarketGameEnv` wraps the same environment for Gymnasium-compatible training
libraries. It maps Gym's `Discrete(3)` action indices as:

```text
0  discharge
1  neutral
2  charge
```

## Feature Reference

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

### `invert_price_to_average_load(price)`

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
if p = 0.10:      M is represented as [-10, 3), estimate = 1.5
if 0.10 < p < .19: M = 3 + (p - 0.10) / 0.03
if 0.19 <= p < .49: M = 6 + (p - 0.19) / 0.10
if 0.49 <= p < 1.49: M = 9 + (p - 0.49) / 0.25
if p >= 1.49:       M = 13 + (p - 1.49)
```

Context: this is the core legal aggregate-observation feature. It reconstructs
a delayed market average, not individual actions.

### `PriceInverse.uncertainty`

Description: width of the feasible inverse interval.

Definition:

```text
uncertainty = inverse_high - inverse_low
```

Context: price `0.10` is lossy because all `M < 3` collapse to the same price.
Policies should be less confident when uncertainty is large or unbounded.

### `estimate_others_average_load(...)`

Description: removes this house's own previous action from the inferred market
average to estimate the average load of the other houses.

Definition:

```text
others_avg_{t-1} = (N * M_{t-1} - L_self,t-1) / (N - 1)
```

Context: this prevents the policy from mistaking its own market impact for
crowd behavior.

### `crowd_delta`

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

### `crowd_battery_belief`

Description: rolling belief over the average battery state of other houses.

Definition:

```text
belief_t = clamp(belief_{t-1} + crowd_delta_{t-1}, 0, battery_capacity)
```

Context: useful for estimating whether the crowd can continue suppressing
future demand. It is approximate because a market average hides distribution.

### `distance_to_nearest_pricing_threshold(average_load)`

Description: measures how close inferred average load is to a pricing tier
boundary.

Definition:

```text
tier_distance = min(|M - b| for b in {3, 6, 9, 13})
```

Context: small distance means a modest change in aggregate load may move the
market into a different price-slope region.

### `recent_mean(values, fallback, window)`

Description: rolling average over recent price history.

Definition:

```text
mean_k = (1 / m) * sum(last m values)
m = min(k, len(values))
```

If no values exist, the feature returns `fallback`.

Context: used as a local reference price for deciding whether current price is
cheap or expensive.

### `recent_volatility(values, window)`

Description: rolling population standard deviation over recent price history.

Definition:

```text
sigma_k = sqrt((1 / m) * sum((x_i - mean_k)^2))
m = min(k, len(values))
```

Context: high volatility can justify less aggressive charging/discharging
because the inferred market state is less stable.

### `future_pressure`

Description: short lookahead average of known future base demand.

Definition:

```text
future_pressure_t = (D_{t+1} + D_{t+2} + D_{t+3}) / 3
```

Context: because the full 24-hour demand profile is passed legally into
`compute_demand`, this helps preserve battery before likely high-demand hours.

### `cheap` And `expensive`

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
