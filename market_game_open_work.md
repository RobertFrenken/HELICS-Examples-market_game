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
- Added extra export-validator negative tests for unsafe imports and runtime
  `eval(...)`.

Validation run for this checkpoint:

```bash
python -m compileall -q python/market_game python/market_game_downstream
python python/market_game/tests/check_all.py
python -m python.market_game_downstream.tests.check_all
python -m python.market_game_downstream.rl.evaluate_scenarios --scenario week_1_baselines --seeds 1,2
git diff --check
```

## Upstream PR Work

Completed or already staged upstream:

- Invalid-demand runtime fix landed upstream as
  `826b5a7 Fix market game invalid demand handling (#134)`.
- Pure-Python simulator/parity branch exists in the clean PR worktree:
  `market-game-simulator-parity`.

Open upstream sequence:

1. Confirm whether the simulator/parity PR has been accepted. If accepted,
   record its PR URL and landed commit here.
2. Port the baseline evaluator next.
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
2. Build aggregate multi-seed summaries from scenario CSV rows. The rows now
   include final battery, clamp count, invalid-load adjustment, invalid-demand
   penalty cost, and price volatility.
3. Add a held-out validation scenario config that is not used for teacher
   action collection.
4. Save evaluation CSVs so runs can be compared without reading terminal logs.
5. Train longer against `week_1_baselines`, then larger scenario configs.
6. Evaluate checkpoints every fixed number of iterations.
7. Compare `local`, `price_history`, and `inference` observation modes.
8. Sweep at least three training seeds before drawing conclusions.

Reference plan: `python/market_game_downstream/rl/docs/rl_tuning_plan.md`.

High-level additions worth doing next:

- Add a scenario result aggregator that reads scenario CSV rows and reports
  mean, standard deviation, min, max, and rank by policy across seed sweeps.
- Add PPO checkpoint multi-seed sweeps matching the fixed-policy
  `evaluate_scenarios --seeds ...` workflow.
- Add a held-out validation scenario config for distillation and final policy
  selection.
- Add one explicit invalid-demand stress scenario so penalty diagnostics are
  exercised in regular evaluation output, not only unit checks.
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
