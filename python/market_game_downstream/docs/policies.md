# Market Game Policy Reference

These policies are local baselines and opponents for the pure simulator. They
all implement the competition-shaped interface:

```python
compute_demand(price, hour, battery_charge, demand, price_history)
```

Unless noted otherwise, they use only legal runtime inputs: current price, hour,
own battery charge, the demand profile, and price history.

## Baselines

| Policy | Strategy | Main use |
|---|---|---|
| `FollowDemandPolicy` | Submits `demand[hour]` and never uses the battery. | Passive reference baseline. |
| `FlattenDemandPolicy` | Charges during below-average base demand and discharges during above-average demand. | Price-blind load-shaping baseline. |
| `FullCyclePolicy` | Charges at the max rate until full, then discharges at the max rate until empty. | Mechanical battery-use baseline and parity check. |
| `PriceAwarePolicy` | Uses fixed cheap/expensive thresholds plus time-of-day reserve targets. | Simple legal heuristic for stock scenarios. |
| `RollingPricePolicy` | Compares current price to a rolling recent mean, with volatility-aware charging. | Legal adaptive heuristic for changing price regimes. |
| `LegalInferencePolicy` | Infers prior average load from price history and adjusts around tier boundaries and expected peaks. | Stronger legal-observation heuristic. |

## Robustness Opponents

| Policy | Strategy | Main use |
|---|---|---|
| `NoisyThresholdPolicy` | Applies seeded jitter to cheap and expensive price thresholds. | Repeatable stochastic opponent. |
| `OscillatingPolicy` | Alternates charge/discharge pressure using a sinusoidal pattern, ignoring price. | Structured non-rational load swings. |
| `VolatilitySeekingPolicy` | Charges into rising prices and discharges once prices are moderate or high. | Chaotic opponent that can amplify market movement. |
| `InvalidDemandPolicy` | Deliberately submits illegal market loads. | Stress-only opponent for clamp, adjustment, and penalty diagnostics. |

## Notes

`boundary_warnings` in simulator output record submitted values outside the
legal battery range. Exact boundary values are valid. `clamps` is usually the
better signal for whether a policy requested a value that changed after
validation.

Exported competition submissions should not import these classes directly. Use
them for local evaluation, then export or hand-write a self-contained
`compute_demand(...)` function under the rules in `rl/export/README.md`.
