# Market Game Export Helpers

This directory is the boundary between flexible local training and a
competition-safe submitted house policy.

The local workbench can use Gymnasium, Ray, Torch, checkpoints, oracle traces,
and scenario randomization. The exported artifact is a standalone `.py` file
with a plain top-level function:

```python
def compute_demand(price, hour, battery_charge, demand, price_history):
    ...
```

Use `validators.py` to check that an exported function has the expected
signature, avoids obviously unsafe runtime dependencies, returns finite numeric
loads, and survives a 24-hour pure-simulator run.

This export format is intentionally narrower than the HELICS classroom template:
do not submit a `House` subclass here. If a local strategy starts as a policy
object or house class, export or copy only its `compute_demand(...)` logic into
the top-level function above.

Examples:

```bash
python3 -m python.market_game_downstream.tests.export_check
```

```python
from python.market_game_downstream.rl.export import validate_submission_file

report = validate_submission_file(
    "python/market_game_downstream/rl/export/example_threshold_submission.py"
)
print(report)
```

`compute_demand_template.py` is intentionally small and self-contained. It is a
starting point for hand-written heuristics or distilled policies, not a training
entry point.

## Distillation

The export path has three distillation targets:

```text
trained PPO checkpoint
  -> teacher action rollouts over named scenarios
  -> fitted student policy
  -> standalone compute_demand file
  -> static and simulator validation
```

Available students:

| Student | Description |
|---|---|
| `threshold` | Small hand-shaped rule with cheap/expensive/late-hour thresholds. |
| `tree` | VIPER-style extracted decision tree policy. Current implementation imitates teacher actions with weighted CART-style splits; full VIPER Q-gap weighting can be added through sample weights. |
| `matrix` | Tiny linear softmax policy trained locally as a NumPy matrix and exported as literal Python lists plus loop-based inference. |

Run threshold distillation with:

```bash
python3 -m python.market_game_downstream.rl.export.distill_checkpoint \
  --checkpoint /tmp/market_game_ppo_week1_seed3_20iter \
  --student threshold \
  --observation-mode price_history \
  --scenario week_1_baselines \
  --scenario week_2_new_profile \
  --scenario week_3_mixed_population \
  --scenario week_4_chaotic_houses \
  --scenario-seed 3 \
  --output /tmp/market_game_distilled_threshold.py
```

Run VIPER-style tree extraction with:

```bash
python3 -m python.market_game_downstream.rl.export.distill_checkpoint \
  --checkpoint /tmp/market_game_ppo_week1_seed3_20iter \
  --student tree \
  --tree-depth 3 \
  --observation-mode price_history \
  --scenario week_1_baselines \
  --scenario week_2_new_profile \
  --scenario week_3_mixed_population \
  --scenario week_4_chaotic_houses \
  --scenario-seed 3 \
  --output /tmp/market_game_distilled_tree.py
```

Run matrix-policy distillation with:

```bash
python3 -m python.market_game_downstream.rl.export.distill_checkpoint \
  --checkpoint /tmp/market_game_ppo_week1_seed3_20iter \
  --student matrix \
  --matrix-epochs 800 \
  --observation-mode price_history \
  --scenario week_1_baselines \
  --scenario week_2_new_profile \
  --scenario week_3_mixed_population \
  --scenario week_4_chaotic_houses \
  --scenario-seed 3 \
  --output /tmp/market_game_distilled_matrix.py
```

Use action-match accuracy as a diagnostic only. A high match rate can mean the
distilled policy captured a strong teacher, but it can also reveal that the
teacher learned a trivial rule such as mostly staying neutral.
