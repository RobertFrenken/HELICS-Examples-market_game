# Market Game RL Architecture

This package has two jobs:

1. Reproduce the HELICS market game rules in a fast pure-Python loop.
2. Provide a legal-observation interface for heuristic and RL agents.

It intentionally stays separate from `python/market_game`, because the original
files are HELICS integration scripts and some execute work at import time.

## Dependency Direction

```text
core/config.py
  -> core/rules.py
  -> core/simulator.py
  -> agents/observations.py
  -> envs/env.py
  -> checks/evaluate.py / checks/parity_check.py / checks/env_check.py
```

`agents/policies.py` depends on `core/config.py` and `agents/features.py`, and
can be used by both `core/simulator.py` and `envs/env.py`.

## Module Roles

| Module | Role |
|---|---|
| `core/config.py` | Episode constants and `MarketGameConfig`. |
| `core/rules.py` | Pure market rules: pricing, clamping, action-to-load conversion. |
| `core/simulator.py` | Full-population episode replay for fixed policies. |
| `core/metrics.py` | Evaluation metrics. |
| `agents/features.py` | Reusable legal feature math. |
| `agents/observations.py` | Named observation schemas and vector builders. |
| `agents/policies.py` | Baseline and heuristic policies. |
| `envs/env.py` | Single-learner, Gymnasium-style environment. |
| `envs/gym_env.py` | Optional Gymnasium adapter around `MarketGameEnv`. |
| `training/train_rllib.py` | Minimal Ray RLlib PPO trainer. |
| `checks/check_all.py` | Unified smoke/check runner. |
| `checks/evaluate.py` | CSV scenario runner. |
| `checks/parity_check.py` | Assertions that pure simulation matches stock HELICS totals. |
| `checks/env_check.py` | Smoke checks for the environment API and observation schemas. |
| `checks/gym_check.py` | Smoke checks for the Gymnasium adapter. |
| `checks/rllib_check.py` | Smoke checks for RLlib PPO integration. |
| `requirements-training.txt` | Optional Gym/Ray/Torch training dependencies. |
| `requirements-helics-local.txt` | Optional HELICS local validation dependencies. |

## Simulator vs Environment

`run_episode()` in `core/simulator.py` runs a complete set of policies against each
other. It is best for parity checks, baseline comparisons, and scenario
evaluation.

`MarketGameEnv` in `envs/env.py` exposes one learner-facing action at a time:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

It is best for RL training and policy search. Hidden aggregate values are
available in `info["diagnostics"]` for debugging, but observation vectors are
constructed only from legal local state, own action history, and delayed price
history.

## Observation Flow

`agents/observations.py` defines named schemas:

- `LOCAL_OBSERVATION_NAMES`
- `PRICE_HISTORY_OBSERVATION_NAMES`
- `INFERENCE_OBSERVATION_NAMES`

The inference observation uses an explicit two-step process:

1. `update_inference_belief(...)` mutates the belief state from delayed price
   and own previous action.
2. `build_inference_observation(...)` converts the current context plus latest
   inference features into a vector.

This split keeps state mutation visible at the environment boundary.

## Rule Sharing

Both `core/simulator.py` and `envs/env.py` call the same `core/rules.py` functions. This keeps
pricing, battery limits, clamping, and discrete battery action semantics aligned
between baseline evaluation and RL training.

## Dependency Tiers

The pure simulator and dependency-free environment should remain usable without
Gymnasium, Ray, Torch, HELICS, or matplotlib.

Gymnasium and Ray/RLlib are offline training layers only. HELICS is used for
local integration validation against the original example. Final deployment
should be distilled back into a self-contained `compute_demand(...)` function.
