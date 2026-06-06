# HELICS Market Game: Next Plan

## Goal

Build a faithful pure-Python training/evaluation path for house-agent strategy
development, then use it to compare heuristics and RL policies before producing
a self-contained `compute_demand(...)` deployment.

## Phase 1: Simulator Parity

Create:

```text
python/market_game_rl/
  simulator.py
  policies.py
  features.py
  evaluate.py
```

First target: reproduce the stock HELICS run using the included example houses.

Expected `profile1` totals:

| Agent | Total cost |
|---|---:|
| `FlattenDemandHouse` | `$35.4467` |
| `FullCycleHouse` | `$47.7467` |
| `PriceAwareHouse` | `$21.9533` |

The simulator must match:

- 24-hour episode length.
- initial price `0.5`;
- `price_history` includes current price;
- one-hour price lag;
- pricing from average market-facing load;
- battery capacity/rate limits;
- market-maker effective clamping;
- negative market-facing load when battery discharge supports it.

## Phase 2: Legal Feature Layer

Implement legal observation/inference helpers:

- price-history features;
- price inversion to previous-hour average load;
- others-only average estimate;
- crowd net battery movement estimate;
- crowd battery belief state;
- tier-boundary risk;
- inversion uncertainty for flat/clipped price regions.

Hidden simulator variables may be used for diagnostics, but not policy inputs.

## Phase 3: Baselines

Implement and evaluate:

- follow-demand;
- flatten-demand clone;
- full-cycle clone;
- price-aware clone;
- rolling-price heuristic;
- legal-inference heuristic.

Evaluation table:

```text
agent, opponent_scenario, mean_cost, std_cost, final_battery, violations, volatility
```

## Phase 4: RL Environment

Wrap the simulator as a Gymnasium-style environment.

Start simple:

- discrete action space: discharge / neutral / charge;
- Tier 1 observation: local state plus price history;
- reward: negative current-hour cost;
- terminal penalty for leftover or depleted battery depending on chosen repeated-day objective.

Then add Tier 2 legal inference observations.

## Phase 5: Robustness Curriculum

Train/evaluate against:

- fixed honest baselines;
- varied threshold policies;
- noisy honest policies;
- mixed honest populations;
- volatility-seeking/adversarial policies;
- randomized house counts and profiles where appropriate.

## Phase 6: Deployment

Produce one self-contained policy file:

- distilled heuristic, or
- tiny embedded neural policy with literal Python weights.

Runtime deployment should avoid external model files, network access, PyTorch,
TensorFlow, Stable-Baselines, and unknown package dependencies.
