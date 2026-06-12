# Market Game Downstream Usage

Run commands from the repo root.

## Checks

Core checks:

```bash
python3 -m python.market_game_downstream.tests.check_all
```

Core checks plus the heavier Ray RLlib smoke check:

```bash
python3 -m python.market_game_downstream.tests.check_all --include-rllib
```

Individual checks:

```bash
python3 -m python.market_game_downstream.tests.shared_core_check
python3 -m python.market_game_downstream.tests.parity_check
python3 -m python.market_game_downstream.tests.env_check
python3 -m python.market_game_downstream.tests.gym_check
python3 -m python.market_game_downstream.tests.rllib_check
```

## Parity Output

```bash
python3 -m python.market_game_downstream.rl.evaluate --stock
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

```bash
python3 -m python.market_game_downstream.rl.evaluate
```

This prints CSV rows for the stock example, all-follow-demand baseline, and
early heuristic mixes. `boundary_warnings` records invalid submitted values.
`clamps` counts effective value changes and is the better constraint-quality
metric.

Invalid demand handling, penalty accounting, adjustment diagnostics, and
combined-bound clamping are documented in `invalid_demand_behavior.md`.

For configurable scenario evaluation with the same diagnostic columns, use:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios
```

Scenario CSV rows include `invalid_load_adjustment`, `penalty_cost`, and
`price_volatility` in addition to the basic cost/load fields.

Use `--seeds 1,2,3` for repeated scenario sweeps.

Save and aggregate a repeated sweep:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios --seeds 1,2,3 > /tmp/scenarios.csv
python3 -m python.market_game_downstream.rl.aggregate_scenarios /tmp/scenarios.csv
```

The aggregate CSV reports mean, sample standard deviation, min, max, and a
per-scenario rank by `total_cost_mean` for each policy.

## Submission Evaluation

Use `evaluate_scenarios --submission` when you want to score the exact
standalone `.py` file intended for submission. This is the current pure-Python
practice workflow; it does not require HELICS and it runs the same
`compute_demand(...)` interface used by the classroom template. The file must
define:

```python
def compute_demand(price, hour, battery_charge, demand, price_history):
    ...
```

Evaluate one submitted file across the default weekly scenarios:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios \
  --submission python/market_game_downstream/rl/export/example_threshold_submission.py \
  --only-submission
```

Evaluate one submitted file against one named scenario:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios \
  --submission python/market_game_downstream/rl/export/example_threshold_submission.py \
  --scenario week_1_baselines \
  --seed 3 \
  --only-submission
```

Omit `--only-submission` to include opponent rows in the CSV. Use `--seeds` or
`--config` the same way as `evaluate_scenarios`. The submission-only rows still
use the scenario CSV format, including `total_load`, `total_cost`,
`final_battery`, `boundary_warnings`, `clamps`, `invalid_load_adjustment`, and
`penalty_cost`.

Pure simulations may also be built directly with `python.market_game_downstream.core`:

```python
from python.market_game_downstream.core import MarketScenario, run_scenario
from python.market_game_downstream.rl.agents.policies import FlattenDemandPolicy, PriceAwarePolicy

scenario = MarketScenario(policies=[FlattenDemandPolicy(), PriceAwarePolicy()])
result = run_scenario(scenario)
```

## Training Smoke Run

Install optional training dependencies only when doing RL experiments:

```bash
python3 -m pip install --user --break-system-packages -r python/market_game_downstream/rl/requirements-training.txt
```

Then run:

```bash
python3 -m python.market_game_downstream.rl.training.train_rllib --iterations 1 --observation-mode price_history
```

The command verifies integration. It is not a tuned experiment.

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

`GymMarketGameEnv` wraps the same environment for Gymnasium-compatible training
libraries. It maps Gym's `Discrete(3)` action indices as:

```text
0  discharge
1  neutral
2  charge
```

Hidden aggregate market values are available only in `info["diagnostics"]`, not
in the observation vector.

## Scenario Builder

Use the Python builder when a JSON file would be tedious to write by hand:

```bash
python3 -m python.market_game_downstream.rl.scenario_builder --output /tmp/custom_scenarios.json
python3 -m python.market_game_downstream.rl.evaluate_scenarios --config /tmp/custom_scenarios.json
```

The builder can describe demand profiles, the controlled RL training agent,
built-in strategy opponents, submitted `compute_demand` functions, stochastic
grab-bag populations, and loaded RL checkpoints. Built-in strategies and
submitted functions run in the pure simulator today; checkpoint players are
recorded as metadata until a checkpoint-opponent wrapper is added.
