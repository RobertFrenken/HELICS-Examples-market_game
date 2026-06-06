# Market Game RL Usage

Run commands from the repo root.

## Dependency Tiers

Core pure simulator:

- Requires only the Python standard library.
- Covers `core/`, `agents/`, the dependency-free `envs/env.py`, and
  `checks/evaluate.py`.

Gymnasium adapter:

- Requires `gymnasium` and `numpy`.
- Install from `requirements-training.txt` if doing RL experiments.

Ray RLlib training:

- Requires Ray/RLlib and Torch.
- Install with:

```bash
python3 -m pip install --user --break-system-packages -r python/market_game_rl/requirements-training.txt
```

Local HELICS validation:

- Requires HELICS and matplotlib.
- Install with:

```bash
python3 -m pip install --user --break-system-packages -r python/market_game_rl/requirements-helics-local.txt
```

Final deployment:

- Should not depend on Gymnasium, Ray, Torch, HELICS, or model checkpoint files.
- The intended final artifact is a self-contained `compute_demand(...)` policy.

## Checks

Run core checks:

```bash
python3 -m python.market_game_rl.checks.check_all
```

Run core checks plus the heavier Ray RLlib smoke check:

```bash
python3 -m python.market_game_rl.checks.check_all --include-rllib
```

## Parity Check

```bash
python3 -m python.market_game_rl.checks.evaluate --stock
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
python3 -m python.market_game_rl.checks.evaluate
```

This prints CSV rows for the stock example, all-follow-demand baseline, and
early heuristic mixes. The `boundary_warnings` column mirrors the original
inclusive boundary warnings. The `clamps` column counts effective value changes
and is the better constraint-quality metric.

Individual checks:

```bash
python3 -m python.market_game_rl.checks.parity_check
python3 -m python.market_game_rl.checks.env_check
python3 -m python.market_game_rl.checks.gym_check
python3 -m python.market_game_rl.checks.rllib_check
```

## Training Smoke Run

```bash
python3 -m python.market_game_rl.training.train_rllib --iterations 1 --observation-mode price_history
```

Ray, RLlib, and Torch are offline-training dependencies only. Deployment should
still use a self-contained `compute_demand(...)` policy with no Ray/Torch
runtime dependency.

The RLlib command is a smoke-training integration test. It confirms that RLlib
can collect complete 24-hour episodes and optimize without API errors. It is
not evidence that the policy has learned a strong strategy yet. Ray 2.55 may
print warnings about its new API stack; those warnings are expected for this
minimal PPO setup.

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

Episode return is the negative learner cost, plus any optional terminal battery
penalty configured for training.
