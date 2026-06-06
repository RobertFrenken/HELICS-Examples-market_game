# Market Game RL Architecture

This package has two jobs:

1. Reproduce the HELICS market game rules in a fast pure-Python loop.
2. Provide a legal-observation interface for heuristic and RL agents.

It intentionally stays separate from `python/market_game`, because the original
files are HELICS integration scripts and some execute work at import time.

## Dependency Direction

```text
config.py
  -> rules.py
  -> simulator.py
  -> observations.py
  -> env.py
  -> evaluate.py / parity_check.py / env_check.py
```

`policies.py` depends on `config.py` and `features.py`, and can be used by both
`simulator.py` and `env.py`.

## Module Roles

| Module | Role |
|---|---|
| `config.py` | Episode constants and `MarketGameConfig`. |
| `rules.py` | Pure market rules: pricing, clamping, action-to-load conversion. |
| `simulator.py` | Full-population episode replay for fixed policies. |
| `features.py` | Reusable legal feature math. |
| `observations.py` | Named observation schemas and vector builders. |
| `env.py` | Single-learner, Gymnasium-style environment. |
| `policies.py` | Baseline and heuristic policies. |
| `metrics.py` | Evaluation metrics. |
| `evaluate.py` | CSV scenario runner. |
| `parity_check.py` | Assertions that pure simulation matches stock HELICS totals. |
| `env_check.py` | Smoke checks for the environment API and observation schemas. |

## Simulator vs Environment

`run_episode()` in `simulator.py` runs a complete set of policies against each
other. It is best for parity checks, baseline comparisons, and scenario
evaluation.

`MarketGameEnv` in `env.py` exposes one learner-facing action at a time:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

It is best for RL training and policy search. Hidden aggregate values are
available in `info["diagnostics"]` for debugging, but observation vectors are
constructed only from legal local state, own action history, and delayed price
history.

## Observation Flow

`observations.py` defines named schemas:

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

Both `simulator.py` and `env.py` call the same `rules.py` functions. This keeps
pricing, battery limits, clamping, and discrete battery action semantics aligned
between baseline evaluation and RL training.
