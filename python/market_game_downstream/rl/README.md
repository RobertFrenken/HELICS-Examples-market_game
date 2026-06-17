# Market Game RL

This directory is the optional reinforcement-learning workbench around the
shared market-game simulator.

Keep it small:

- `agents/`: hand-authored baseline policies and legal observation features.
- `envs/`: dependency-free environment plus the optional Gymnasium adapter.
- `training/`: RLlib training entry point.
- `scenarios.py`: built-in evaluation/training scenarios.
- `evaluate.py`: the one CSV evaluator for scenarios and standalone
  `compute_demand(...)` submissions.
- `export/`: validation and policy-to-function adapters for standalone
  submissions.

Run the default scenario curriculum:

```bash
python3 -m python.market_game_downstream.rl.evaluate
```

Evaluate one scenario:

```bash
python3 -m python.market_game_downstream.rl.evaluate --scenario week_1_baselines
```

Evaluate a standalone submission:

```bash
python3 -m python.market_game_downstream.rl.evaluate \
  --scenario week_1_baselines \
  --submission python/market_game_downstream/rl/export/example_threshold_submission.py \
  --only-submission \
  --validate-submission
```

Train PPO:

```bash
python3 -m python.market_game_downstream.rl.training.train_rllib \
  --scenario week_1_baselines
```
