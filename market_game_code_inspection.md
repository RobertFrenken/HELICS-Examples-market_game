# HELICS Market Game: Code Inspection Notes

This note records the executable mechanics in `python/market_game` that matter
for a pure simulator, RL environment, and legal deployment policy.

## Player Hook

House strategies subclass `House` and implement:

```python
compute_demand(price, hour, battery_charge, demand, price_history)
```

The method is called once per hour from `House.run()`.

Legal direct inputs:

- `price`: current published market price.
- `hour`: integer hour `0..23`.
- `battery_charge`: local battery state before this hour's action.
- `demand`: the full 24-hour base demand profile assigned to this house.
- `price_history`: prices observed so far, including the current hour's price.

The returned value is the house's market-facing load for the current hour.
Relative to `base = demand[hour]`:

- `return base` leaves the battery unchanged.
- `return base + delta` charges by `delta`.
- `return base - delta` discharges by `delta`.

## Market Timing

The initial price is `0.5`.

At each hour:

1. Houses read the current price.
2. Houses publish current-hour market-facing load.
3. The market maker reads all current-hour loads.
4. The market maker charges current-hour cost as `current_price * load`.
5. The market maker computes the next price from current-hour aggregate load.
6. The next price is published for the next hour.

Therefore, the useful inference alignment is:

```text
price_history[i + 1] = pricing_rule(average_market_load_at_hour_i)
```

The exception is `price_history[0] == 0.5`, which is the exogenous initial
price and does not invert to hour `-1` load.

## Pricing Rule

The market maker computes:

```python
M = total_load / number_of_house_federates
```

Then:

| Average load `M` | Price |
|---:|---:|
| `M < 3.0` | `0.10` |
| `3.0 <= M < 6.0` | `0.10 + 0.03 * (M - 3.0)` |
| `6.0 <= M < 9.0` | `0.19 + 0.10 * (M - 6.0)` |
| `9.0 <= M < 13.0` | `0.49 + 0.25 * (M - 9.0)` |
| `M >= 13.0` | `1.49 + 1.00 * (M - 13.0)` |

Inversion notes:

- `price == 0.10` only implies `M < 3.0`; use an interval or a conservative
  estimate rather than an exact point.
- For `price > 0.10`, the piecewise linear regions are invertible.
- Boundary prices are ambiguous in principle, but the same average estimate is
  usually fine for control features.

## Demand Profiles

The market maker assigns each house a profile during initialization.

Profiles:

- `flat`: `[5] * 24` because non-flat update is skipped.
- default/fallback: `[5] * 24`.
- `profile1`: `[2, 1, 1, 1, 2, 4, 6, 8, 9, 7, 5, 4, 3, 4, 5, 7, 9, 12, 10, 7, 5, 4, 2, 2]`.
- `random`: random 24-hour profile scaled to total `120`.
- `spike`: mostly `4`, one random hour at `28`.
- `dspike`: mostly `3`, two random added spikes of `24`.
- `profile_solar`: a fixed profile scaled by `3.0`, including negative base
  demand during solar hours.

Under the default `run_neighborhood.py --profile profile1`, all houses share
the same `profile1` base demand.

## Battery Rules

Constants:

- Capacity: `20.0` kWh.
- Initial charge: `0.0` kWh.
- Max charge rate: `5.0` kWh/hour.
- Max discharge rate: `10.0` kWh/hour.

Battery state updates with:

```python
battery.change(market_load - base_demand)
```

Positive delta charges. Negative delta discharges.

Both the house template and market maker validate/clamp submitted load. For an
RL simulator, match the market maker's final effective clamp, because that is
what determines price and cost.

Important nuance: the clamp does not enforce non-negative market load. If the
battery has enough energy, a house may legally return a negative load, and the
market maker will include that negative value in aggregate load and cost.

## Number Of Houses

`run_neighborhood.py` discovers house files matching `*_house.py` in the
selected folder. It creates one broker, one federate per matching house file,
and one market maker. The market maker then counts all federates except itself
as houses.

For pure simulation, make `num_houses` explicit in config and use the same
average-load pricing rule.

## Current Example Policies

Included baseline strategies:

- `house_test.py`: simple alternate behavior, charges by `2` if empty and
  otherwise discharges by `2`.
- `flatten_demand_house.py`: ignores price and tries to move load toward the
  daily average.
- `full_cycle_house.py`: ignores price and alternates max charging to full and
  max discharging to empty.
- `price_aware_house.py`: threshold policy that charges at cheap prices,
  preserves reserve by hour, discharges at high prices, and drains late-day
  leftover charge.

These are good initial opponents and baseline evaluation agents.

## Simulator Requirements

A faithful pure simulator should:

- start hour `0` with `current_price = 0.5`;
- pass `price_history` including the current price to policy functions;
- compute current-hour cost with the current price before updating price;
- update next price from current-hour average effective market load;
- allow negative market load when battery discharge supports it;
- duplicate the market maker clamp semantics;
- support shared `profile1` first, then randomized profiles;
- expose hidden aggregate variables only in `info`/diagnostics, not policy
  observations.

## RL Handoff Adjustments

The existing RL handoff is directionally correct, with these code-derived
corrections:

- Do not clamp deployed/simulated purchase to `max(0.0, purchase)` unless the
  contest rules add a no-export constraint. Current code allows negative load.
- Treat `price_history[-1]` as the current price, not the previous price.
- For aggregate inversion at hour `t`, use `price_history[t]` to infer hour
  `t-1` load only when `t > 0`.
- The initial price `0.5` is exogenous and should be a reset/config parameter.
- Match the validation boundaries exactly if reproducing printed warnings
  matters; for control/economics, the effective clamped value is what matters.
