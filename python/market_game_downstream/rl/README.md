# Market Game RL Helpers

This package provides the fast offline path for the HELICS market game:

- shared pure-Python market rules from `python/market_game_downstream/core`
- baseline policies
- weekly scenario and chaotic-house scaffolding
- a dependency-free Gymnasium-style environment
- optional Gymnasium/RLlib adapters
- export validators for self-contained `compute_demand(...)` submissions
- parity and smoke tests

Use this package to train and evaluate strategies quickly. Final submissions
should still be self-contained house policies with no Ray, Torch, Gymnasium, or
checkpoint dependency.

## Quick Commands

Run core checks:

```bash
python3 -m python.market_game_downstream.tests.check_all
```

Evaluate the stock example policies:

```bash
python3 -m python.market_game_downstream.rl.evaluate --stock
```

Evaluate the weekly scenario curriculum:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios
```

Evaluate one named scenario from the default JSON scenario registry:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios --scenario week_3_mixed_population --seed 7
```

Validate the example exported submission:

```bash
python3 -m python.market_game_downstream.tests.export_check
```

Run a short PPO smoke-training job:

```bash
python3 -m python.market_game_downstream.rl.training.train_rllib --iterations 1 --observation-mode price_history
```

Train PPO against the same named opponent scenario used by CSV evaluation:

```bash
python3 -m python.market_game_downstream.rl.training.train_rllib \
  --iterations 1 \
  --observation-mode price_history \
  --scenario week_3_mixed_population \
  --scenario-seed 7
```

Train against a larger population with non-smoke PPO settings:

```bash
python3 -m python.market_game_downstream.rl.training.train_rllib \
  --iterations 500 \
  --observation-mode price_history \
  --scenario-config python/market_game_downstream/rl/scenario_configs/large_population.json \
  --scenario large_random_grab_bag_40 \
  --scenario-seed 11 \
  --train-batch-size 4096 \
  --minibatch-size 256 \
  --num-epochs 10 \
  --checkpoint-dir /tmp/market_game_ppo_large_random_seed11
```

Use `price_history` as the default RL observation mode. The `inference` mode is
available for advanced experiments and is documented in `docs/reference/features.md`.

## Scenario Configs

In this package, a scenario means a repeatable experiment configuration:
demand profile, random seed, and opponent population. This matches common
multi-agent RL usage where scenario names identify benchmark/game variants,
while the Gymnasium/RLlib object is still the environment.

The default registry is `scenario_configs/weekly.json`. Larger populations are
in `scenario_configs/large_population.json`. These files use standard JSON so
scenario loading has no YAML dependency. Opponents can be listed by policy
class name or as objects with `type`, `kwargs`, and optional `count`; stochastic
policy seeds can use `$seed`, `$seed+N`, `$seed-N`, `$index`, or
`$seed+$index` placeholders.

For the full authoring format, including stochastic `grab_bag` populations, see
`docs/scenario_config_schema.md`.

## Dependency Tiers

| Tier | Requires | Use |
|---|---|---|
| Shared core | Python standard library | Rules, simulation, tests. |
| Gym adapter | `gymnasium`, `numpy` | Library-compatible env wrapper. |
| RLlib training | Ray/RLlib, Torch | Offline training experiments. |
| HELICS validation | HELICS, matplotlib | Local integration against the original game. |

Install training dependencies only when needed:

```bash
python3 -m pip install --user --break-system-packages -r python/market_game_downstream/rl/requirements-training.txt
```

## Read Next

- `../docs/usage.md`: command reference and expected parity output.
- `../docs/policies.md`: baseline and opponent policy behavior.
- `../docs/competition_workbench.md`: legal observation boundaries and weekly scenarios.
- `docs/rl_training.md`: simulator-vs-HELICS workflow and deployment path.
- `docs/scenario_config_schema.md`: scenario config format for exact and grab-bag populations.
- `docs/scenario_training_and_distillation.md`: scenario-driven PPO, checkpoint evaluation, and export distillation.
- `../docs/architecture.md`: package boundaries after the shared-core refactor.
- `../docs/upstream_pr_path.md`: staged path toward smaller upstream PRs.
- `docs/reference/features.md`: observation and inference feature math.
- `export/README.md`: submission wrapper and validation boundary.
