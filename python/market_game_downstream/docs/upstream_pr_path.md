# Upstream PR Path

Use the downstream package to harden ideas before proposing small changes to
the upstream `market_game` branch.

Recommended sequence:

1. Pure Python market rules and parity checks.
2. Baseline scenario evaluator.
3. Optional weekly profile/opponent scenario configuration.
4. Gymnasium adapter.
5. Optional RL training examples.
6. Deployment/export examples for self-contained `compute_demand(...)`.

Each PR should keep the official HELICS runtime behavior unchanged unless the
PR is specifically about runtime integration.

The first PR should also carry the invalid-demand documentation and clamp-order
fix described in `invalid_demand_behavior.md`. That issue affects the
dependency-free rules layer and should be resolved before presenting the
simulator as the authoritative validation path.

Good first upstream pitch:

> Add optional training/evaluation tools for the market-game competition:
> a dependency-free simulator, parity checks against the stock houses, baseline
> scenarios, and documentation that keeps submitted policies limited to the
> legal `compute_demand(...)` interface.

Avoid bundling heavy dependencies, trained models, or broad HELICS refactors in
the first upstream PR.
