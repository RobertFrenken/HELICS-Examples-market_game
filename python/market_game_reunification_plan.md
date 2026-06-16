# Market Game Reunification Plan

Goal: reunify `market_game_downstream` toward the official `market_game`
competition code while preserving two downstream goals:

- a pure-Python route to simulate and evaluate strategies without HELICS;
- reinforcement learning as an optional way to author better strategies.

The consolidation should be submission-first. The official user-facing strategy
surface remains:

```python
def compute_demand(price, hour, battery_charge, demand, price_history):
    ...
```

Everything else should help users produce, test, or export that function.

## Direction

Do not treat downstream as a replacement fork of `market_game`. Treat it as a
workbench that proves ideas before they move into upstream-shaped paths.

The stable center is already present:

- `market_game/house_template.py` teaches the `compute_demand(...)` interface.
- `market_game_downstream/core/simulator.py` runs that same interface without
  HELICS.
- `market_game_downstream/rl/export` validates and emits standalone
  `compute_demand(...)` submissions.

The consolidation should reduce duplicate rule implementations, not add more
wrappers around them.

## Current Status

Completed local workflow pieces:

- duplicate-rule inventory: `market_game_downstream/docs/rule_inventory.md`;
- compact standalone-submission scoring:
  `python3 -m python.market_game_downstream.rl.evaluate_submission`;
- richer scenario scoring for standalone files through
  `evaluate_scenarios --submission`;
- held-out validation scenarios:
  `market_game_downstream/rl/scenario_configs/held_out_validation.json`;
- invalid-demand stress scenario:
  `market_game_downstream/rl/scenario_configs/invalid_demand_stress.json`;
- routine smoke coverage for generated threshold, tree, and matrix exports and
  for invalid-demand penalty diagnostics.

Still open: upstream simulator/parity PR #136 acceptance, small
upstream-shaped evaluator ports, PPO checkpoint sweep tooling, and
checkpoint-as-peer opponent support.

Retired in this repo: the duplicate `market_game_downstream.helics` runtime
copy. The remaining files under that directory are compatibility notes pointing
to canonical `python/market_game`.

## Target Roles

| Area | Role | Dependency rule |
|---|---|---|
| `market_game` | Official classroom and competition-facing code. | Keep the runtime shape familiar to upstream users. |
| `market_game/simulation` | Optional pure-Python simulator and evaluator, if accepted upstream. | Standard library only. |
| `market_game_downstream/core` | Local canonical pure rule engine until pieces are ported upstream. | Standard library only. |
| `market_game_downstream/rl` | Training, scenario generation, evaluation, and diagnostics. | Optional RL dependencies stay here. |
| `market_game_downstream/rl/export` | Boundary from local training to standalone submission files. | Exported files must be self-contained. |

## Non-Negotiables

1. `compute_demand(...)` is the competition ABI.
2. HELICS is not required for local strategy development.
3. RL does not create a second runtime interface.
4. Pure simulator behavior must stay covered by parity checks.
5. Core rule code must stay dependency-free.
6. Exported submissions must not depend on Ray, Torch, Gymnasium, checkpoints,
   scenario files, file reads, or network access.
7. Do not add new user-facing strategy interfaces. `choose_action`,
   `choose_delta`, policy objects, and RL actions may exist internally or
   downstream, but they must compile down to `compute_demand(...)` and should
   not be promoted as competition APIs.

## Staged Plan

### 1. Freeze The Contract

Write one concise contract for `compute_demand(...)` and make every path point
to it:

- HELICS house template;
- pure simulator policy protocol;
- baseline policies;
- submitted-file runner;
- export validator;
- RL distillation output.

Acceptance check:

- a plain standalone file defining `compute_demand(...)` can be validated and
  scored in the pure simulator.

### 2. Inventory Duplicate Rule Behavior

Create a short mapping of where each rule currently lives:

- battery capacity, charge, and discharge limits;
- valid demand clamping;
- invalid demand warnings and penalties;
- price calculation;
- hourly market stepping;
- stock demand profiles.

For each rule, name the desired canonical implementation.

Acceptance check:

- the inventory identifies which downstream code should be ported, which code
  should wrap shared behavior, and which code should be left alone.

### 3. Prove The Submission-File Path

Before porting richer evaluator or scenario features, prove the smallest useful
pure-Python workflow: score one standalone file that defines only
`compute_demand(...)`.

Downstream already has this shape through:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios --submission path/to/submission.py
```

First, decide whether that existing route should become the documented workflow
as-is or whether it needs a very thin alias with less RL/scenario vocabulary.
Do not write a second evaluator until the current route has been judged too
heavy for users.

Move only the dependency-free pieces needed for that path:

- price calculation;
- battery/clamp helpers;
- one-hour market step;
- a tiny 24-hour runner;
- focused parity tests.

Leave stock baseline evaluators and richer scenario support downstream until the
submission-file command is working. Do not upstream the whole
`market_game_downstream` package.

Acceptance check:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios --submission path/to/submission.py --only-submission
python3 python/market_game/tests/check_all.py
python3 -m python.market_game_downstream.tests.check_all
```

Expected evaluator shape:

```text
house,total_load,total_cost,final_battery,clamps
```

### 4. Promote The Submission Workflow

After the minimum submission-file path is proven, make it the default practice
workflow for users. This is documentation and command ergonomics first, not a
new technical layer.

Users should be able to write a submission-shaped function, run it locally,
and inspect results without starting HELICS.

The pure path should answer:

- What did my house load each hour?
- What did it cost?
- Did it violate battery rules?
- How did it compare to baseline strategies?

Acceptance check:

- one documented command evaluates a standalone `compute_demand(...)` file;
- optional baseline comparison is available without requiring users to learn
  RL training concepts.

### 5. Keep RL Behind Export

RL is a training method, not a user-facing runtime requirement.

Do not upstream `ActionHouse`, `DeltaHouse`, RL action spaces, checkpoint
wrappers, or policy object APIs as user-facing competition surfaces.

The RL path should be:

```text
train or tune policy
  -> evaluate across scenarios
  -> distill or export
  -> standalone compute_demand.py
  -> validate
  -> score in pure simulator
```

Acceptance check:

- a distilled or hand-authored export passes validation and scenario scoring
  without importing downstream helper modules.

### 6. Thin Downstream After Upstream-Shaped Code Exists

Once `market_game/simulation` has stable pieces, downstream should either import
them or clearly wrap them. Avoid maintaining two independent simulators.

Acceptance check:

- downstream tests still pass;
- the source of truth for each core rule is obvious from the code, not only
  from docs.

## Guardrails For Future Work

Each session should produce one concrete artifact:

- one parity test;
- one dependency-free module port;
- one evaluator command;
- one exported submission check;
- one deletion or simplification of duplicated rule code;
- one concise doc update that reflects code already in place.

Avoid broad rewrites, new helper layers, and large markdown-only planning
sessions. If a change does not improve parity, exportability, or dependency
separation, it is probably not on the critical path.

## Immediate Next Steps

1. Wait for upstream simulator/parity PR #136 to land, then record the landed
   commit in `market_game_open_work.md`.
2. Prepare the smallest dependency-free simulator/evaluator PR needed by the
   submission workflow.
3. Add focused parity tests for any ported rules.
4. Add PPO checkpoint sweep tooling and checkpoint-by-checkpoint validation CSV
   output.
5. Add a checkpoint-as-peer opponent wrapper only if checkpoint policies need
   to participate as ordinary pure-simulator players.
6. Thin downstream imports only after upstream-shaped code exists.
