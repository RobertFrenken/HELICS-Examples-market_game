# Competition Workbench

This downstream package is a workbench for designing market-game strategies
before submitting a house to the competition VM.

The official submission surface remains:

```python
compute_demand(price, hour, battery_charge, demand, price_history)
```

Everything in this package should help teams produce a better implementation
of that function. It should not require changes to the official game runtime.

## Data Boundaries

Use these names consistently:

| Name | Available at deployment? | Purpose |
|---|---:|---|
| Legal observation | Yes | Inputs available to `compute_demand(...)`. |
| Inference feature | Yes, if derived only from legal history | Price-history and delayed aggregate estimates. |
| Diagnostics | No | Simulator metrics for debugging and evaluation. |
| Oracle frame | No | Hidden simulator state for labels, plots, and analysis. |

RL training may use diagnostics or oracle frames for auxiliary losses,
imitation labels, plots, or curriculum design. A submitted policy must not need
that data at runtime.

## Weekly Scenarios

`python.market_game_downstream.rl.scenarios` provides small scenario builders
for local evaluation:

- `stock_example_scenario()`: parity scenario for the current example houses.
- `weekly_training_scenarios(seed=...)`: baseline, new-profile, mixed, and
  chaotic-house curricula.
- `evaluate_curriculum(seed=...)`: CSV-friendly summary rows.

The weekly helpers are intentionally lightweight. They are not predictions of
the official competition schedule; they are local stress tests for strategies.

## Chaotic Houses

The downstream baseline policies are summarized in `policies.md`. They include
opponents that are deliberately less stable than the stock examples:

- `NoisyThresholdPolicy`: price-aware behavior with deterministic threshold
  jitter.
- `OscillatingPolicy`: predictable charge/discharge swings.
- `VolatilitySeekingPolicy`: pushes harder when prices are near higher tiers.

These opponents are useful for robustness testing and should not be treated as
official competition agents.

## Deployment Goal

The preferred output of training is a self-contained house policy:

- distilled heuristic, or
- small embedded model with Python literal weights.

Avoid runtime dependencies on Ray, Torch, Gymnasium, checkpoints, external
files, or network access unless the competition rules explicitly allow them.
