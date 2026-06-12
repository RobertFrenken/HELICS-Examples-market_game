# Market Game Rule Inventory

This inventory tracks which implementation owns each market-game rule while the
downstream workbench is being reunified with the official `market_game` package.
The current direction is submission-first: user-authored strategies should stay
plain `compute_demand(...)` functions, and duplicate rule code should shrink
around that interface.

## Canonical Sources

| Rule | Current locations | Desired canonical implementation | Session decision |
|---|---|---|---|
| Battery capacity and charge/discharge limits | `market_game/battery.py`, `market_game_downstream/core/config.py`, `market_game_downstream/core/simulator.py` | Downstream canonical: `market_game_downstream.core`; upstream target: future `market_game/simulation` if accepted. | Keep `market_game/battery.py` as official runtime behavior for now. Port only dependency-free equivalents in a later simulator PR. |
| Valid demand clamping | `market_game/battery.py`, `market_game_downstream/core/rules.py` | `market_game_downstream/core/rules.py::clamp_market_load` until an upstream pure simulator exists. | Do not add more wrappers. Existing `ensure_valid` compatibility helpers should continue to delegate to `clamp_market_load`. |
| Invalid demand warnings and penalties | `market_game/battery.py`, `market_game/market_maker.py`, `market_game_downstream/core/rules.py`, `market_game_downstream/core/simulator.py` | Clamping/warnings in `core.rules`; penalty accounting in the hourly transition. | Keep HELICS runtime code stable. Use parity checks before moving this upstream. |
| Price calculation | `market_game/market_maker.py`, `market_game_downstream/core/rules.py` | `compute_price_from_average_load` and `compute_price_from_total_load` in `core.rules`. | Later upstream PR should port only this dependency-free function plus focused tests. |
| Hourly market stepping | `market_game/market_maker.py`, `market_game_downstream/core/simulator.py` | `market_game_downstream/core/simulator.py::step_market_hour` until upstream-shaped simulator code exists. | Treat HELICS `run_market_hour` as the official runtime adapter, not the reusable rule engine. |
| Stock demand profiles | `market_game/market_maker.py`, `market_game_downstream/core/config.py` | `market_game_downstream/core/config.py::demand_profile`. | Keep profile behavior aligned by parity tests; do not move scenario curriculum upstream yet. |

## Wrapper Classification

`market_game_downstream/rl/core/rules.py` and
`market_game_downstream/rl/core/simulator.py` are compatibility re-exports.
They are downstream-only wrappers, not canonical rule implementations. New code
should import from `python.market_game_downstream.core`.

`market_game_downstream/rl/agents` may keep policy objects, action helpers, and
training conveniences for experiments. Those are not competition ABI surfaces
and should not be promoted into `market_game` as user-facing strategy APIs.

`market_game_downstream/rl/export` is the boundary from downstream training or
experimentation back to the competition ABI. Exported files must define
`compute_demand(...)` and must not import downstream helpers.

## Submission Workflow Check

The promoted compact route scores one standalone submission file:

```bash
python3 -m python.market_game_downstream.rl.evaluate_submission \
  python/market_game_downstream/rl/export/example_threshold_submission.py
```

It reuses the scenario evaluator and emits:

```text
house,total_load,total_cost,final_battery,clamps
```

The existing downstream scenario route can also score a standalone submission
file:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios \
  --submission python/market_game_downstream/rl/export/example_threshold_submission.py \
  --only-submission
```

Do not build a second simulation path unless the existing evaluator proves too
heavy for users.

Current output is scenario-oriented CSV with these columns:

```text
scenario,agent,profile_type,seed,total_load,total_cost,final_battery,boundary_warnings,clamps,invalid_load_adjustment,penalty_cost,price_volatility
```

A later ergonomics commit may add optional baseline comparison to the compact
practice route.
