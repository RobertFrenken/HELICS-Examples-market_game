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

Validate the example exported submission:

```bash
python3 -m python.market_game_downstream.tests.export_check
```

Run a short PPO smoke-training job:

```bash
python3 -m python.market_game_downstream.rl.training.train_rllib --iterations 1 --observation-mode price_history
```

Use `price_history` as the default RL observation mode. The `inference` mode is
available for advanced experiments and is documented in `docs/reference/features.md`.

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
- `../docs/architecture.md`: package boundaries after the shared-core refactor.
- `../docs/upstream_pr_path.md`: staged path toward smaller upstream PRs.
- `docs/reference/features.md`: observation and inference feature math.
- `export/README.md`: submission wrapper and validation boundary.
