# Market Game Downstream

This directory is a downstream peer to `python/market_game`.

Use `python/market_game` as the upstream-facing HELICS example. Keep broader
experiments here so upstream updates can be merged without mixing local RL and
refactor work into the original example directory.

Working fork: <https://github.com/RobertFrenken/HELICS-Examples-market_game>

Layout:

- `core/`: dependency-free market rules and pure Python simulator.
- `docs/`: downstream architecture, usage, competition, and upstreaming notes.
- `helics/`: compatibility notes for the retired downstream HELICS mirror.
- `tests/`: smoke, parity, scenario, and export validation tests.
- `rl/`: policies, observation builders, Gymnasium/RLlib adapters, and training/evaluation commands.
- `rl/export/`: wrappers and validators for self-contained `compute_demand`
  submissions.
- `rl/scenario_configs/`: weekly, large-population, held-out validation, and
  invalid-demand stress scenario JSON files.

Run the local checks from the repository root:

```bash
python3 -m python.market_game_downstream.tests.check_all
```

Read next:

- `docs/competition_workbench.md`: legal observations vs diagnostics/oracle
  data, weekly scenarios, and deployment goals.
- `docs/policies.md`: baseline and opponent policy behavior.
- `docs/upstream_pr_path.md`: proposed upstream PR sequence, starting with the
  invalid-demand runtime fix and then moving to simulator/evaluation/RL/export
  pieces.
