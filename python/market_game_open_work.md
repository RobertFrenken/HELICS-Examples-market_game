# Market Game Open Work

This is the single root-level tracker for unfinished market-game work. Completed
handoff notes, rule references, and design background have been consolidated
into maintained docs under `python/market_game` and
`python/market_game_downstream`.

## Current Context

- Comprehensive repo: `/home/rober/HELICS-Examples-market_game`
- Clean upstream PR worktree:
  `/home/rober/HELICS-Examples-market_game-pr-invalid-demand`
- Upstream target: `GMLC-TDC/HELICS-Examples:market_game`
- Keep upstream-facing example work under `python/market_game`.
- Keep broader experiments under `python/market_game_downstream`.

Run the downstream smoke suite from the comprehensive repo root:

```bash
python3 -m python.market_game_downstream.tests.check_all
```

Expected checks:

```text
shared core: ok
import safety: ok
stock parity: ok
env smoke: ok
scenario smoke: ok
export smoke: ok
gym env smoke: ok
helics config: ok
```

## Recent Cleanup And Robustness Changes

This checkpoint consolidated the repo and hardened the active market-game
paths:

- Replaced the root `market_game_*.md` handoff/planning files with this single
  open-work tracker.
- Cleaned root and market-game markdown links, including removal of
  machine-specific absolute paths.
- Refactored `python/market_game/market_maker.py` into import-safe helper
  functions with a `main()` guard.
- Added upstream-facing market-game checks:
  `python/market_game/tests/check_all.py` and
  `python/market_game/tests/check_market_maker.py`.
- Tidied the original market-game battery/template/example-house files while
  preserving backward compatibility for the misspelled `BATTERY_CAPCITY`
  symbol.
- Fixed downstream HELICS `DeltaHouse` validation to use shared helpers that
  accept numeric battery charge.
- Updated downstream invalid-demand/upstream-path docs to reflect the accepted
  invalid-demand PR.
- Hardened RL scenario loading:
  - malformed top-level scenario JSON is rejected;
  - unknown policies and malformed seed placeholders are covered by checks;
  - duplicate policy names in one scenario are rejected before diagnostics can
    collapse by name.
- Expanded scenario CSV diagnostics with `invalid_load_adjustment`,
  `penalty_cost`, and `price_volatility`.
- Added fixed-policy multi-seed scenario sweeps with
  `python3 -m python.market_game_downstream.rl.evaluate_scenarios --seeds 1,2,3`.
- Added a scenario result aggregator:
  `python3 -m python.market_game_downstream.rl.aggregate_scenarios`.
- Added a scenario builder for demand profiles, controlled RL training agents,
  built-in strategy populations, submitted function opponents, and loaded RL
  checkpoint metadata:
  `python3 -m python.market_game_downstream.rl.scenario_builder`.
- Added extra export-validator negative tests for unsafe imports and runtime
  `eval(...)`.
- Added a HELICS runner-config generator that converts downstream RL scenarios
  into canonical `python/market_game` federations and can insert a standalone
  `compute_demand(...)` export as the first house.
- Added compact standalone-submission scoring:
  `python3 -m python.market_game_downstream.rl.evaluate_submission`.
- Added `python/market_game_downstream/docs/rule_inventory.md` to identify
  canonical rule owners and downstream-only wrappers.
- Added generated-submission regression checks for threshold, tree, and matrix
  distillation exports.
- Added held-out validation scenarios in
  `python/market_game_downstream/rl/scenario_configs/held_out_validation.json`.
- Added invalid-demand stress scenarios in
  `python/market_game_downstream/rl/scenario_configs/invalid_demand_stress.json`
  and routine smoke coverage that exercises penalty diagnostics.
- Confirmed upstream simulator/parity PR status on 2026-06-16:
  <https://github.com/GMLC-TDC/HELICS-Examples/pull/136> is open, not merged.
- Retired the duplicate downstream HELICS runtime files under
  `python/market_game_downstream/helics`; remaining files are compatibility
  notes pointing to canonical `python/market_game`.

Validation run for this checkpoint:

```bash
python -m compileall -q python/market_game python/market_game_downstream
python python/market_game/tests/check_all.py
python -m python.market_game_downstream.tests.check_all
python -m python.market_game_downstream.rl.evaluate_scenarios --scenario week_1_baselines --seeds 1,2
python -m python.market_game_downstream.rl.helics_config --scenario week_1_baselines --output /tmp/houses.json
git diff --check
```

## Upstream PR Work

Completed or already staged upstream:

- Invalid-demand runtime fix landed upstream as
  `826b5a7 Fix market game invalid demand handling (#134)`.
- Pure-Python simulator/parity PR is open, not merged:
  <https://github.com/GMLC-TDC/HELICS-Examples/pull/136>
  (`market-game-simulator-parity`).

Open upstream sequence:

1. Wait for the simulator/parity PR to be accepted. If accepted, record its
   landed commit here.
2. Port the baseline evaluator next after PR #136 settles.
   - Source candidates:
     `python/market_game_downstream/rl/agents/policies.py`,
     `python/market_game_downstream/rl/evaluate.py`, and small shared pieces
     from `python/market_game_downstream/rl/core/`.
   - Upstream-shaped destination:
     `python/market_game/simulation/policies.py`,
     `python/market_game/simulation/evaluate.py`,
     `python/market_game/tests/check_simulation_evaluate.py`, and
     `python/market_game/docs/simulation.md`.
   - Keep it dependency-free.
   - Expected CLI:
     `python3 -m python.market_game.simulation.evaluate`
   - Expected output columns:
     `house,total_load,total_cost,final_battery,clamps`
3. Later PRs, only after earlier ones settle:
   - scenario configs;
   - local submitted-policy runner or strategy workbench;
   - lightweight legal feature helpers;
   - optional Gymnasium adapter;
   - submission/export validator;
   - RL training examples only if maintainers want them.

Upstream rule: do not upstream `python/market_game_downstream` directly. Port
small, reviewable pieces into upstream-shaped paths under `python/market_game`.

## Downstream Training Work

The downstream simulator, baseline policies, scenario configs, Gymnasium/RLlib
plumbing, and export validators exist. PPO is still a pipeline baseline, not a
tuned competitive agent.

Open training/evaluation tasks:

1. Add multi-seed scenario sweeps for PPO checkpoints. Fixed-policy scenario
   sweeps are available through `evaluate_scenarios --seeds ...`.
2. Add an executable RLlib checkpoint opponent wrapper if loaded checkpoint
   agents need to participate as peer players inside pure-simulator scenario
   evaluations. The builder currently records loaded checkpoints as metadata.
3. Save evaluation CSVs so runs can be compared without reading terminal logs.
4. Train longer against `week_1_baselines`, then larger scenario configs.
5. Evaluate checkpoints every fixed number of iterations.
6. Compare `local`, `price_history`, and `inference` observation modes.
7. Sweep at least three training seeds before drawing conclusions.

Reference plan: `python/market_game_downstream/rl/docs/rl_tuning_plan.md`.

High-level additions worth doing next:

- Add PPO checkpoint multi-seed sweeps matching the fixed-policy
  `evaluate_scenarios --seeds ...` workflow.
- Use generated HELICS configs as the final local validation step for distilled
  policies after simulator evaluation and before classroom/CTF handoff.
- Add a true loaded-checkpoint player wrapper if checkpoint policies should run
  as market peers rather than only as the controlled RL learner or distillation
  teacher.
- Use `held_out_validation.json` for distillation and final policy selection.
- Keep `invalid_demand_stress.json` in routine smoke checks so penalty
  diagnostics stay exercised in regular evaluation output, not only unit
  checks.
- Decide whether `rl/evaluate.py` should remain as the fixed-baseline
  compatibility command or become a thin alias around `evaluate_scenarios`.

## Distillation And Export Work

The export layer exists, but the current distilled students mainly prove the
pipeline. Stronger training and validation are still needed before choosing a
competition submission.

Open export tasks:

1. Distill the best PPO checkpoint into threshold, tree, and matrix students.
2. Add a tiny embedded neural-net student only if the matrix student is too
   weak.
3. Evaluate distilled students on held-out scenarios against
   `PriceAwarePolicy`, `RollingPricePolicy`, and `LegalInferencePolicy`.
4. Select exports by held-out market performance, not action-match accuracy
   alone.
5. Tighten validator checks when VM rules become clear:
   - allowed imports;
   - file access;
   - helper functions;
   - source size;
   - runtime/time limits.
6. Keep final submission artifacts self-contained:
   - no Ray, Torch, Gymnasium, checkpoint, model-file, file-read, or network
     dependency;
   - deterministic unless randomness is explicitly allowed and seeded;
   - reset state cleanly at hour `0`;
   - pass `validate_submission_file(...)`.

Reference docs:

- `python/market_game_downstream/rl/export/README.md`
- `python/market_game_downstream/rl/docs/scenario_training_and_distillation.md`

## Submission-First Refactor Plan

Goal: reshape the downstream competition workbench closer to the official
`python/market_game` form while retaining RL as a local training and export
pipeline. The submitted artifact should be a single `.py` file whose runtime
surface is:

```python
def compute_demand(price, hour, battery_charge, demand, price_history):
    ...
```

This is the boundary to optimize around. HELICS classes, downstream helper
imports, Gymnasium, Ray, Torch, checkpoints, and scenario files can exist in the
training workspace, but they should not be required by the final CTF submission.

Target architecture:

| Layer | Target role | Dependency rule |
|---|---|---|
| `python/market_game` | Official classroom/CTF runtime shape, examples, docs, HELICS runner. | HELICS/matplotlib are runtime conveniences, not submission requirements. |
| `python/market_game/simulation` | Upstream-shaped pure simulator and baseline evaluator, if accepted by upstream. | Standard library only. |
| `python/market_game_downstream/core` | Local canonical rule engine until pieces are ported upstream. | Standard library only. |
| `python/market_game_downstream/rl` | Training, scenario generation, evaluation, distillation, and diagnostics. | Optional RL dependencies stay here. |
| `python/market_game_downstream/rl/export` | Hard boundary that renders and validates standalone submissions. | Output files must be self-contained. |

Refactor phases:

1. Preserve the official house form as the canonical submission interface.
   - Treat `compute_demand(...)` as the only stable competition ABI.
   - Keep `ActionHouse` and `DeltaHouse` as teaching/workbench conveniences,
     but do not rely on them in exported submissions.
   - Make examples and docs say clearly that helper classes are local
     authoring tools, while submission is a plain function or plain subclass
     method body.

2. Move shared pure-game behavior toward upstream-shaped modules.
   - Port small dependency-free pieces from `market_game_downstream.core` into
     `python/market_game/simulation` when ready for upstream review.
   - Keep the port narrow: price calculation, battery clamping, scenario
     stepping, baseline policies, evaluator CLI, and focused tests.
   - Avoid direct upstreaming of `market_game_downstream`; use it as the source
     workbench only.

3. Make export the first-class RL deliverable.
   - The RL agent can use rich observations and dependencies during training.
   - Distillation should produce threshold, tree, or matrix students as literal
     Python source with `compute_demand(...)`.
   - Add a local command that validates an exported file against the same
     signature, import, finite-output, clamp, and 24-hour simulation checks used
     by `validate_submission_file(...)`.

4. Add a submission harness that can run the exact submitted file locally.
   - Load `compute_demand(...)` from a path.
   - Wrap it with `FunctionSubmissionPolicy`.
   - Evaluate it against fixed, large-population, held-out, and invalid-demand
     stress scenarios.
   - Emit CSV rows with total cost, load, final battery, clamps, penalty cost,
     and opponent set.

5. Tighten the validator only after VM rules are known.
   - Current allowed-import posture is deliberately narrow.
   - Keep source-size, import, top-level execution, runtime-call, signature, and
     simulator checks separate so rule changes are easy to apply.
   - Prefer false negatives during competition hardening over silently allowing
     dependencies that will fail in the VM.

6. Keep RL competitive work downstream.
   - Train/evaluate PPO or other agents in `market_game_downstream/rl`.
   - Select policies by held-out market performance, not only action-match
     accuracy.
   - Distill the selected policy into standalone source and validate that source
     as the final artifact.

Near-term implementation checklist:

1. Done: added `--submission PATH` support to
   `python.market_game_downstream.rl.evaluate_scenarios`, reusing the existing
   scenario/evaluation plumbing directly.
2. Done: added `held_out_validation.json` for validation scenarios not used by
   the default weekly teacher-collection examples.
3. Done: added `invalid_demand_stress.json` and smoke coverage for invalid-load
   adjustment and penalty diagnostics.
4. Done: added generated-submission regression checks for threshold, tree, and
   matrix exports.
5. Done: documented the two supported local authoring paths:
   - hand-written `compute_demand(...)`;
   - RL checkpoint -> distilled standalone `compute_demand(...)`.
6. When upstream simulator parity settles, port the evaluator/baseline pieces
   into `python/market_game/simulation` in small PR-sized commits.

Course correction:

The `--submission` evaluator support is useful, but it is not the main
structural refactor. Treat it as workflow plumbing, not the simplifying move.
The more important refactor is to reduce the number of competing house/runtime
forms.

Current risk:

- `python/market_game` has the official classroom and CTF-facing house form.
- `python/market_game_downstream/helics` has a second HELICS-facing house,
  battery, market-maker, and documentation surface.
- RL/export code targets `compute_demand(...)`, but the duplicated HELICS layer
  makes it unclear which house form is canonical.

Corrected direction:

1. Make `python/market_game` the canonical house/runtime shape for submission
   and local HELICS play.
2. Keep `python/market_game_downstream` as the simulator, training,
   evaluation, validation, and export workbench.
3. Done: retired the duplicate `python/market_game_downstream/helics` runtime
   files. Compatibility notes now point back to `python/market_game`.
4. Move any genuinely useful downstream HELICS improvements, such as
   dependency-light imports or action/delta authoring helpers, toward the
   canonical `python/market_game` path only if they do not make submission
   rules less clear.
5. Prefer refactors that remove duplicated concepts or code paths over new
   commands that wrap existing behavior.

Next real refactor candidate:

After upstream simulator/parity PR #136 settles, port the dependency-free
baseline evaluator into upstream-shaped `python/market_game/simulation` paths.
Keep the port small and avoid upstreaming downstream RL/scenario machinery.

## Documentation Cleanup Still Open

The root planning notes were consolidated here. Remaining documentation cleanup
should happen in the maintained locations:

- `python/market_game/market_game.md`
- `python/market_game/house_strategy_tutorial.md`
- `python/market_game_downstream/README.md`
- `python/market_game_downstream/docs/`
- `python/market_game_downstream/rl/docs/`

Open doc tasks:

1. Update this file after each upstream PR with PR URL, branch, landed commit,
   accepted status, and changed next step.
2. Keep official-game docs focused on gameplay and submission rules.
3. Keep downstream docs focused on training, evaluation, export, and local
   experimentation.

## References

Use these maintained docs instead of restoring old root handoff files:

- `python/market_game/market_game.md`
- `python/market_game_downstream/docs/architecture.md`
- `python/market_game_downstream/docs/competition_workbench.md`
- `python/market_game_downstream/docs/invalid_demand_behavior.md`
- `python/market_game_downstream/docs/policies.md`
- `python/market_game_downstream/docs/upstream_pr_path.md`
- `python/market_game_downstream/docs/usage.md`
- `python/market_game_downstream/helics/docs/game_rules.md`
- `python/market_game_downstream/rl/docs/reference/features.md`
- `python/market_game_downstream/rl/docs/rl_training.md`
- `python/market_game_downstream/rl/docs/rl_tuning_plan.md`
- `python/market_game_downstream/rl/docs/scenario_config_schema.md`
- `python/market_game_downstream/rl/docs/scenario_training_and_distillation.md`
