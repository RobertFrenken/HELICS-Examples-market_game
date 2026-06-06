# Market Game RL Helpers

Pure-Python simulator, baseline agents, Gymnasium/RLlib training adapters, and
checks for the HELICS market game.

This package intentionally does not import from `python/market_game`. The
original market-game files are HELICS integration scripts; this package is the
fast offline simulation/training path.

## Start Here

- [docs/usage.md](docs/usage.md): commands, dependency tiers, and checks.
- [docs/architecture.md](docs/architecture.md): package boundaries and module roles.
- [docs/features.md](docs/features.md): observation and inference feature math.

## Quick Commands

Run core checks:

```bash
python3 -m python.market_game_rl.checks.check_all
```

Run checks including the heavier Ray RLlib smoke test:

```bash
python3 -m python.market_game_rl.checks.check_all --include-rllib
```

Run a short PPO smoke-training job:

```bash
python3 -m python.market_game_rl.training.train_rllib --iterations 1 --observation-mode price_history
```

## Layout

```text
core/      market rules, simulator, metrics
agents/    baseline policies and legal feature builders
envs/      dependency-free env plus optional Gymnasium adapter
training/  offline RL training entry points
checks/    parity, smoke, and scenario checks
docs/      detailed package documentation
```

Final deployment should still be distilled into a self-contained
`compute_demand(...)` policy with no Gym/Ray/Torch runtime dependency.
