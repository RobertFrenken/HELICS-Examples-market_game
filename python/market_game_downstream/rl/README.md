# Market Game RL

This directory is the optional reinforcement-learning workbench around the
shared market-game simulator.

Keep it small:

- `agents/`: percepts, semantic actions, controllers, action projectors,
  learner action spaces, and legal observation features.
- `envs/`: dependency-free environment, scenario builders, and the optional
  Gymnasium adapter.
- `training/`: reward shaping, RLlib primitives, and the training CLI.
- `export/`: export profiles, validation, and policy-to-function adapters for
  standalone submissions.
- `evaluate.py`: the one CSV evaluator for scenarios and standalone
  `compute_demand(...)` submissions.

## Agent Grammar

The agent refactor uses a controller-centered grammar:

```text
legal percept + state/belief -> controller -> semantic market action -> market load
```

Preset `compute_demand(...)` houses remain supported as simulator adapters over
controllers. Agent code uses `MarketAgent`, `MarketPercept`, controller
classes, semantic market actions, and `project_market_load`. RL policies fit
as learned controllers in `agents/controllers/learned.py`: they own feature
extractors, vector models, and action decoders internally, while training
algorithms and reward shaping stay in `training/`.

Controllers that need memory receive it through `MarketAgent.state`. Inference
controllers and inference feature extractors use `InferenceBeliefState`; static
controllers use `NoAgentState`.

See `agents/REFRACTOR.md` for the target end state and migration plan.

Install training dependencies for the downstream RL workflow:

```bash
uv sync --extra training
```

If you also need the HELICS runtime in the same environment, select both extras:

```bash
uv sync --extra training --extra helics
```

List the built-in scenario IDs:

```bash
python3 -m python.market_game_downstream.rl.evaluate --list-scenarios
```

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

## RLlib Training Modes

Train, export, validate, and evaluate a standalone competition submission:

```bash
python3 -m python.market_game_downstream.rl.training.train \
  --mode exportable \
  --scenario week_1_baselines \
  --output runs/rl_exportable/submission.py \
  --evaluate-scenario week_1_baselines \
  --evaluate-scenario week_2_new_profile
```

Or keep those settings in a TOML file:

```bash
python3 -m python.market_game_downstream.rl.training.train \
  --config python/market_game_downstream/rl/training/exportable.example.toml
```

This command uses the supported export profile: `price_history` observations,
the discrete battery-posture action space, and a one-hidden-layer tanh actor
with 8 units. Arbitrary RLlib PPO checkpoints are not assumed to be
competition-exportable.

In TOML, `iterations` is PPO training iterations. `episodes_per_iteration`
controls how many 24-hour market-game episodes are sampled per iteration.
The default reward is negative hourly cost. `final_battery_target` and
`final_battery_penalty` add optional terminal reward shaping for training only.

Use the same entry point for smoke checks or unconstrained experiments:

```bash
python3 -m python.market_game_downstream.rl.training.train \
  --mode smoke \
  --scenario week_1_baselines \
  --iterations 1
```

```bash
python3 -m python.market_game_downstream.rl.training.train \
  --mode experiment \
  --scenario week_1_baselines \
  --iterations 5 \
  --checkpoint-dir runs/rl_experiment/checkpoints \
  --evaluate-scenario week_1_baselines
```
