# Downstream Usage

Run retained downstream checks:

```bash
python3 -m python.market_game_downstream.tests.check_all
```

Evaluate the default weekly RL scenarios:

```bash
python3 -m python.market_game_downstream.rl.evaluate
```

Evaluate one scenario or a seed sweep:

```bash
python3 -m python.market_game_downstream.rl.evaluate --scenario week_1_baselines
python3 -m python.market_game_downstream.rl.evaluate --scenario week_1_baselines --seeds 1,2,3
```

Evaluate a standalone `compute_demand(...)` submission:

```bash
python3 -m python.market_game_downstream.rl.evaluate \
  --scenario week_1_baselines \
  --submission python/market_game_downstream/rl/export/example_threshold_submission.py \
  --only-submission \
  --validate-submission
```

Train with RLlib:

```bash
python3 -m python.market_game_downstream.rl.training.train \
  --mode experiment \
  --scenario week_1_baselines
```

Shared rule code lives in `python.market_game_downstream.core`. Built-in RL
scenarios live in `python.market_game_downstream.rl.envs.scenarios`; add
custom experiments there or construct `CompetitionScenario` objects in Python.
