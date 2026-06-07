# Market Game RL Helpers

This package provides the fast offline path for the HELICS market game:

- shared pure-Python market rules from `python/market_game_downstream/core`
- baseline policies
- a dependency-free Gymnasium-style environment
- optional Gymnasium/RLlib adapters
- parity and smoke checks

Use this package to train and evaluate strategies quickly. Final submissions
should still be self-contained house policies with no Ray, Torch, Gymnasium, or
checkpoint dependency.

## Quick Commands

Run core checks:

```bash
python3 -m python.market_game_downstream.rl.checks.check_all
```

Evaluate the stock example policies:

```bash
python3 -m python.market_game_downstream.rl.checks.evaluate --stock
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
| Shared core | Python standard library | Rules, simulation, checks. |
| Gym adapter | `gymnasium`, `numpy` | Library-compatible env wrapper. |
| RLlib training | Ray/RLlib, Torch | Offline training experiments. |
| HELICS validation | HELICS, matplotlib | Local integration against the original game. |

Install training dependencies only when needed:

```bash
python3 -m pip install --user --break-system-packages -r python/market_game_downstream/rl/requirements-training.txt
```

## Read Next

- `docs/usage.md`: command reference and expected parity output.
- `docs/rl_training.md`: simulator-vs-HELICS workflow and deployment path.
- `docs/architecture.md`: package boundaries after the shared-core refactor.
- `docs/reference/features.md`: observation and inference feature math.
