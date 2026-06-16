# Invalid Demand Behavior, Penalty, and Clamp Bug

This behavior is implemented in the downstream simulator and was the basis for
the accepted upstream invalid-demand runtime fix. The downstream simulator
clamps with combined legal bounds so validation always returns a battery-safe
market load.

## Current Intended Behavior

When a policy returns an impossible market load, the submitted value is clamped
back into the legal battery range, the policy receives the effective clamped
outcome, and an explicit invalid-demand penalty is added to the clamped energy
cost:

```text
proposed_market_load
  -> clamp to battery constraints
  -> market_load
  -> energy_cost = price * market_load
  -> penalty_cost = 20 * abs(proposed_market_load - market_load)
  -> cost = energy_cost + penalty_cost
```

The simulator records two diagnostics:

- `boundary_warnings`: warning strings from the original validation helper.
- `clamps`: count of hours where `market_load != proposed_market_load`.
- `invalid_load_adjustment`: absolute difference between proposed and effective
  market load.
- `penalty_cost`: dollar penalty for invalid submissions.

`clamps` is the better signal for whether the policy actually requested a value
that changed after validation. Exact legal boundary values are valid and do not
emit warnings.

## Charging Past Capacity

If a house tries to buy more than the battery can store, the market load is
clamped to the remaining battery capacity.

Example:

```text
base_demand = 2
battery_charge = 19
capacity = 20
proposed_market_load = 100
```

Effective result:

```text
market_load = 3
battery_charge becomes 20
cost = price * 3
invalid_load_adjustment = abs(100 - 3)
penalty_cost = 20 * invalid_load_adjustment
warning = listed battery charge rate exceeds available battery storage capacity
```

The house pays for the clamped effective market load plus the invalid-demand
penalty.

## Discharging Without Energy

If a house tries to serve demand from battery energy it does not have, the
market load is clamped back up to what the battery can actually support.

Example:

```text
base_demand = 2
battery_charge = 0
proposed_market_load = -100
```

Effective result:

```text
market_load = 2
battery_charge stays 0
cost = price * 2
invalid_load_adjustment = abs(-100 - 2)
penalty_cost = 20 * invalid_load_adjustment
warning = listed consumption exceeds available battery energy
```

Again, the house gets no credit for impossible discharge and pays the
invalid-demand penalty.

## Upstream Penalty Fix

The original upstream market maker computed:

```text
penaltyCost = 20 * abs(load - valid_load)
```

but did not add that value to `fed.hourCost`; the recorded hourly cost was still
`load * current_price` after clamping. The accepted upstream fix adds
`penaltyCost` to `fed.hourCost` so invalid submissions are strictly worse than
valid boundary actions.

## HELICS Log Message

The canonical HELICS runtime in `python/market_game` prints invalid-demand
messages ending with:

```text
and assessing penalty=<penalty_cost>
```

The HELICS `hourCost` total includes the clamped energy cost and the
invalid-demand penalty. The older downstream HELICS runtime mirror has been
retired so this behavior has one HELICS implementation.

## Fixed Clamp-Order Bug

The original helper checked capacity/available-energy limits before
charge/discharge-rate limits:

```python
if market_load >= base_demand + remaining_capacity:
    ...
if market_load >= base_demand + max_charge:
    ...
if market_load <= base_demand - battery_charge:
    ...
if market_load <= base_demand - max_discharge:
    ...
```

That order could return a clamped value that still violated the hourly rate
limit when a proposed value violated both constraints.

Example over-charge case:

```text
base_demand = 2
battery_charge = 0
remaining_capacity = 20
max_charge = 5
proposed_market_load = 100
```

The old first capacity clamp returned:

```text
market_load = 22
delta = +20
```

That still exceeded the max charge rate of `5`, so the subsequent battery
update could raise:

```text
ValueError: requested charge rate exceeds maximum rate
```

Example over-discharge case:

```text
base_demand = 12
battery_charge = 20
max_discharge = 10
proposed_market_load = -100
```

The old available-energy clamp returned:

```text
market_load = -8
delta = -20
```

That still exceeded the max discharge rate of `10`, so the battery update could
raise:

```text
ValueError: requested discharge exceeds maximum discharge rate
```

## Implemented Fix

Compute the legal lower and upper market-load bounds by combining all
constraints, then clamp once:

```text
upper = base_demand + min(max_charge, capacity - battery_charge)
lower = base_demand - min(max_discharge, battery_charge)
market_load = min(max(proposed_market_load, lower), upper)
```

The warning can then describe the violated constraint, but the returned
`market_load` should always be safe for `BatteryState.change(...)`.

The downstream checks include direct tests for:

- over capacity near full battery
- over charge rate with empty battery
- over discharge with empty battery
- over discharge rate with charged battery
- exact-boundary behavior, so valid boundary values are intentionally documented
