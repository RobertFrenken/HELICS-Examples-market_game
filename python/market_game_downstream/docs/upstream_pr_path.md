# Upstream PR Path

Use the downstream package to harden ideas before proposing small changes to
the upstream `market_game` branch.

Working fork:

- <https://github.com/RobertFrenken/HELICS-Examples-market_game>

Recommended PR sequence:

1. Runtime rule fix: safe invalid-demand clamping plus explicit invalid-demand
   penalty accounting in the official `python/market_game` runtime.
2. Pure Python market rules and parity checks.
3. Baseline scenario evaluator.
4. Optional weekly profile/opponent scenario configuration.
5. Gymnasium adapter.
6. Optional RL training examples.
7. Deployment/export examples for self-contained `compute_demand(...)`.

Each PR after the runtime rule fix should keep the official HELICS runtime
behavior unchanged unless the PR is specifically about runtime integration.

The first PR should carry only the invalid-demand documentation, combined-bound
clamp behavior, explicit invalid-demand penalty, and focused battery-validation
checks described in `invalid_demand_behavior.md`.

Good first upstream pitch:

> Make invalid market demand handling safe and enforce the documented penalty:
> clamp submitted demand against combined battery bounds, treat exact boundary
> values as valid, add the invalid-demand penalty to scored cost, and cover the
> edge cases with focused checks.

Good second upstream pitch:

> Add optional training/evaluation tools for the market-game competition:
> a dependency-free simulator, parity checks against the stock houses, baseline
> scenarios, and documentation that keeps submitted policies limited to the
> legal `compute_demand(...)` interface.

Avoid bundling heavy dependencies, trained models, broad HELICS refactors, or
the downstream RL workbench in the first upstream PR.

## Proposed Future PRs

| PR | Scope | Notes |
|---|---|---|
| 1 | Invalid-demand runtime fix | `battery.py`, `market_maker.py`, and focused validation checks only. |
| 2 | Dependency-free simulator | Shared rules, stock parity checks, and documentation for legal policy inputs. |
| 3 | Baseline evaluator | CSV scenario evaluation for stock and simple heuristic policies. |
| 4 | Scenario curriculum | Weekly/new-profile/mixed-population/chaotic-opponent configs. |
| 5 | Gymnasium adapter | Optional environment wrapper without mandatory training dependencies. |
| 6 | RL examples | Optional RLlib training smoke examples and docs. |
| 7 | Export path | Self-contained `compute_demand(...)` validators and examples. |
