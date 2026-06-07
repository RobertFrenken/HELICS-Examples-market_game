# Market Game Downstream Handoff

## Context

This repo is a fork of `GMLC-TDC/HELICS-Examples`, focused on the
`market_game` branch. The upstream-facing HELICS example should remain in
`python/market_game`. Broader experiments now live in the peer directory:

```text
python/market_game_downstream/
```

The motivation is a multi-week capture-the-flag style market-game competition.
Teams submit house strategy code to a VM. Valid houses compete against other
team houses. The winner of week `t` has its `compute_demand(...)` function
opened to all teams for week `t + 1`. Later weeks may introduce new demand
profiles and more chaotic/opponent houses.

The long-term goal is to make strategy development, baseline comparison, and
optional RL training accessible to other teams while keeping official gameplay
and submission rules clear.

## Current Design

Downstream layout:

```text
python/market_game_downstream/
  core/
    config.py
    rules.py
    simulator.py
  docs/
    architecture.md
    competition_workbench.md
    invalid_demand_behavior.md
    policies.md
    upstream_pr_path.md
    usage.md
  helics/
    battery.py
    house_template.py
    market_maker.py
    market_game.md
    docs/game_rules.md
  tests/
    check_all.py
    *_check.py
  rl/
    agents/
    core/
    docs/
    envs/
    export/
    training/
    evaluate.py
    evaluate_scenarios.py
    scenarios.py
```

Key idea:

- `python/market_game` stays close to upstream.
- `market_game_downstream/core` contains dependency-free market rules and a
  pure Python simulator.
- `market_game_downstream/helics` is a downstream HELICS mirror that can use
  the shared core.
- `market_game_downstream/rl` contains optional policies, observations,
  scenarios, Gymnasium/RLlib adapters, export helpers, and evaluation commands.
- `market_game_downstream/tests` contains smoke, parity, scenario, and export
  validation tests.

## Legal Observation Boundary

The official deployed strategy interface is:

```python
compute_demand(price, hour, battery_charge, demand, price_history)
```

Important distinction:

- Legal observations: exactly the inputs available to `compute_demand(...)`,
  plus features derived only from those inputs/history.
- Diagnostics: simulator metrics useful for debugging and evaluation.
- Oracle data: hidden simulator state useful for training labels, plots,
  supervised auxiliary losses, or analysis.

Training code may use diagnostics/oracle data, but a submitted competition
policy must not require them at runtime.

The downstream simulator now includes:

- `LegalHouseInput`
- `OracleMarketFrame`
- `SimulationResult.oracle_trace()`

These names are intended to make the boundary explicit.

## Scenario/Competition Workbench

`python.market_game_downstream.rl.scenarios` provides:

- `CompetitionScenario`
- `stock_example_scenario()`
- `weekly_training_scenarios(seed=...)`
- `evaluate_scenario(...)`
- `evaluate_curriculum(...)`

Scenario CSV evaluation now exists:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios
```

The weekly scenarios are local stress tests, not official schedules. They cover
baseline, new-profile, mixed-population, and chaotic-house cases.

Additional robustness opponents currently include:

- `NoisyThresholdPolicy`
- `OscillatingPolicy`
- `VolatilitySeekingPolicy`

Policy behavior is documented both in class docstrings and in:

```text
python/market_game_downstream/docs/policies.md
```

Invalid demand behavior and the combined-bound clamp behavior are
documented in:

```text
python/market_game_downstream/docs/invalid_demand_behavior.md
```

That note is marked as first-PR material because invalid-demand accounting
affects the shared rules/simulator layer.

## Export Layer Direction

Do not force the whole workbench to obey VM submission restrictions. Keep
training flexible. A first export/submission layer now exists:

```text
python/market_game_downstream/rl/export/
  README.md
  compute_demand_template.py
  example_threshold_submission.py
  export_policy.py
  validators.py
```

Training side can use Gymnasium, Ray, Torch, checkpoints, oracle diagnostics,
plots, and scenario randomization.

Export side should produce/check a competition-safe submission:

- exposes `compute_demand(price, hour, battery_charge, demand, price_history)`
- avoids disallowed runtime imports
- avoids network/file dependencies unless allowed
- can embed distilled heuristics or small model weights as Python literals
- returns numeric market demand
- survives a 24-hour simulation
- behaves legally under battery constraints after market-maker clamping

Current validator behavior:

- checks the exact `compute_demand(...)` signature
- caps source-file size for submitted source validation
- rejects obvious unsafe imports/runtime calls in source-file submissions
- rejects import-time side effects and non-literal top-level assignments
- checks finite numeric return values
- wraps a function as a simulator policy
- runs a 24-hour pure-simulator smoke check against simple opponents

This is the preferred framing:

> Teams can train or design strategies however they want locally, and the
> workbench helps validate/export a legal `compute_demand(...)` submission.

## Similar Project Conventions

The current design follows conventions from:

- Grid2Op / L2RPN: power-grid competition environment with observations,
  actions, rewards, scenarios/time-series, opponents, runners, and Gym wrappers.
- EV2Gym: V2G/EV charging simulator with Gym-style RL interface and benchmark
  framing.
- PowerTAC: open-source power trading agent competition.
- ASSUME: agent-based electricity market simulator with RL integration.
- FinRL / Gym trading environments: market data/scenario layer, Gym-style
  environment, baselines, evaluation/backtesting, optional training stack.

Common pattern:

```text
domain simulator
  -> Gymnasium-style environment wrapper
  -> configurable scenarios/data
  -> baselines
  -> optional RL training scripts
  -> evaluation/backtesting
  -> deployment/export path
```

The market-game competition is stricter than many Gym projects because the VM
may only accept a submitted function. That means the export layer matters.

## Upstream PR Strategy

For higher acceptance upstream, do not start with the full downstream workbench.
Propose small, reviewable PRs:

1. Dependency-free market rules/simulator plus parity checks.
2. Baseline scenario evaluator.
3. Optional weekly profile/opponent scenario configuration.
4. Gymnasium adapter.
5. Optional RL training examples.
6. Export/deployment examples for self-contained `compute_demand(...)`.

Pitch as:

> Optional training/evaluation tools for the market-game competition.

Avoid pitching the first PR as:

> Add RL to the official game.

The official runtime should remain unchanged unless a PR is explicitly about
runtime integration.

## Current Verification

Run from repo root:

```bash
python3 -m python.market_game_downstream.tests.check_all
```

Expected output includes:

```text
shared core: ok
import safety: ok
stock parity: ok
env smoke: ok
scenario smoke: ok
export smoke: ok
gym env smoke: ok
```

## Recent Git State

The downstream peer-package refactor was committed and pushed to the fork:

```text
c75b6fd Move market game experiments downstream
```

Remote layout:

- `origin`: `https://github.com/RobertFrenken/HELICS-Examples-market_game.git`
- `upstream`: `https://github.com/GMLC-TDC/HELICS-Examples.git`

Normal pushes to `origin` go to the fork, not upstream.

After that commit, additional downstream changes were made but not yet
committed at the time this handoff was written:

- legal/oracle dataclasses
- scenario curriculum helpers
- chaotic opponent policies
- scenario smoke check
- export/submission wrapper and validators
- competition workbench docs
- upstream PR path docs

Check `git status --short` before continuing.

## Suggested Next Steps

1. Decide whether to commit the current downstream additions as a checkpoint.
2. Consider adding stricter export checks if the VM rules become clearer
   (allowed imports, file access, helper functions, time limits).
3. Once downstream stabilizes, extract PR 1 as a clean branch from
   `upstream/market_game`.
