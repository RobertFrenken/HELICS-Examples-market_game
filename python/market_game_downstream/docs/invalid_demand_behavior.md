# Invalid Demand Behavior and Clamp Bug

**First PR candidate:** this behavior should be documented and tested in one of
the first upstream PRs, ideally with the dependency-free market rules/simulator
and parity checks. The clamp-order edge case should be fixed before relying on
the simulator as the authoritative validation path.

## Current Intended Behavior

When a policy returns an impossible market load, the market maker does not add
an explicit dollar penalty. The submitted value is clamped back into the legal
battery range, the policy receives the effective clamped outcome, and cost is
computed from that clamped market load:

```text
proposed_market_load
  -> clamp to battery constraints
  -> market_load
  -> cost = price * market_load
```

The simulator records two diagnostics:

- `boundary_warnings`: warning strings from the original validation helper.
- `clamps`: count of hours where `market_load != proposed_market_load`.

`clamps` is the better signal for whether the policy actually requested a value
that changed after validation. `boundary_warnings` mirror the original
inclusive helper behavior and can be emitted at exact legal boundaries.

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
warning = listed battery charge rate exceeds available battery storage capacity
```

There is no extra implemented penalty beyond losing the impossible extra
charge. The house pays only for the clamped effective market load.

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
warning = listed consumption exceeds available battery energy
```

Again, there is no extra implemented dollar penalty. The policy simply does not
get credit for impossible discharge.

## Misleading HELICS Log Message

The downstream HELICS mirror currently prints a message ending with:

```text
and assessing penalty
```

No such penalty is implemented in the shared simulator path or the downstream
HELICS market-maker accounting. That message should be corrected or paired with
an actual explicit penalty rule.

## Known Clamp-Order Bug

The current clamp function checks capacity/available-energy limits before
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

That order can return a clamped value that still violates the hourly rate limit
when a proposed value violates both constraints.

Example over-charge case:

```text
base_demand = 2
battery_charge = 0
remaining_capacity = 20
max_charge = 5
proposed_market_load = 100
```

The first capacity clamp returns:

```text
market_load = 22
delta = +20
```

That still exceeds the max charge rate of `5`, so the subsequent battery update
can raise:

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

The available-energy clamp returns:

```text
market_load = -8
delta = -20
```

That still exceeds the max discharge rate of `10`, so the battery update can
raise:

```text
ValueError: requested discharge exceeds maximum discharge rate
```

## Recommended Fix

Compute the legal lower and upper market-load bounds by combining all
constraints, then clamp once:

```text
upper = base_demand + min(max_charge, capacity - battery_charge)
lower = base_demand - min(max_discharge, battery_charge)
market_load = min(max(proposed_market_load, lower), upper)
```

The warning can then describe the violated constraint, but the returned
`market_load` should always be safe for `BatteryState.change(...)`.

The first PR should include direct tests for:

- over capacity near full battery
- over charge rate with empty battery
- over discharge with empty battery
- over discharge rate with charged battery
- exact-boundary behavior, so warnings vs clamps are intentionally documented
