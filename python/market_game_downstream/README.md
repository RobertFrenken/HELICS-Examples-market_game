# Market Game Downstream

This directory is a downstream peer to `python/market_game`.

Use `python/market_game` as the upstream-facing HELICS example. Keep broader
experiments here so upstream updates can be merged without mixing local RL and
refactor work into the original example directory.

Layout:

- `core/`: dependency-free market rules and pure Python simulator.
- `helics/`: downstream HELICS mirror that can import the shared core.
- `rl/`: policies, observation builders, Gymnasium/RLlib adapters, and checks.

Run the local checks from the repository root:

```bash
python3 -m python.market_game_downstream.rl.checks.check_all
```
