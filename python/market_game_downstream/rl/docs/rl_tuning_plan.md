# RL Tuning and Distillation Plan

This plan starts from the current state: the simulator, Gymnasium wrapper,
RLlib smoke-training path, scenario configs, and distillation/export validators
exist, but PPO is still a pipeline baseline rather than a tuned competitive
agent.

## Phase 1: Evaluation Harness

Goal: make policy comparisons repeatable before spending time on longer
training runs.

Tasks:

- Use `evaluate_scenarios --seeds ...` for fixed-policy multi-seed sweeps, and
  add the same checkpoint sweep behavior for PPO.
- Build multi-seed summaries from scenario CSV rows, which already include
  final battery, clamp count, invalid-load adjustment, invalid-demand penalty
  cost, and price volatility.
- Support a validation scenario config that is not used for teacher-action
  collection.
- Save CSV output so runs can be compared without reading terminal logs.

Acceptance checks:

- The same policy and seed set produces deterministic CSV rows.
- `PriceAwarePolicy`, `RollingPricePolicy`, and `LegalInferencePolicy` are
  included as standard comparison baselines.
- Invalid-load adjustment diagnostics and penalty costs appear separately from
  energy costs.

## Phase 2: PPO Training Runs

Goal: establish whether PPO can beat the strongest hand-written baselines under
the legal observation modes.

Tasks:

- Train longer against `week_1_baselines`, then against the larger scenario
  config.
- Evaluate checkpoints every fixed number of iterations rather than only at the
  end.
- Compare `local`, `price_history`, and `inference` observation modes.
- Sweep at least three training seeds before drawing conclusions.
- Track train scenario cost and held-out validation cost separately.

Acceptance checks:

- Checkpoint-by-checkpoint validation CSV identifies the best checkpoint by
  market performance, not by imitation accuracy.
- At least one PPO run is compared against the standard heuristic baselines on
  held-out scenarios.

## Phase 3: Distillation

Goal: turn the best available teacher into a self-contained
`compute_demand(...)` submission that remains auditable.

Tasks:

- Distill the best PPO checkpoint into the existing threshold, tree, and matrix
  students.
- Add a tiny embedded neural-net student only if the matrix student is too weak.
- Evaluate distilled students on held-out scenarios and standard baselines.
- Record both action-match accuracy and market-performance metrics.

Acceptance checks:

- Exported files pass `validate_submission_file(...)`.
- Distilled policy validation cost is reported against the same baselines used
  for PPO checkpoints.
- The chosen export is selected by held-out market performance, not only action
  match.

## Phase 4: Recommendation

Goal: produce a defensible submission candidate.

Tasks:

- Compare best hand-written heuristic, best PPO checkpoint, and best distilled
  policy across the same multi-seed held-out sweep.
- Document which observations the final policy uses.
- Keep the final artifact self-contained and free of runtime training,
  checkpoints, heavy ML imports, file reads, and network access.

Acceptance checks:

- Final recommendation includes cost table, residual risks, and validation
  command.
- Submission candidate survives a 24-hour simulator smoke test with zero
  effective clamps unless deliberate invalid-demand testing is being run.
