# Scenario Training and Distillation

This document explains the current RL workbench for the downstream market game:
how scenarios are defined, how PPO is trained and evaluated, and how a trained
policy can be distilled into a competition-safe `compute_demand(...)` function.

## Short Version

The workflow is:

```text
scenario config
  -> PPO training
  -> checkpoint evaluation
  -> teacher action collection
  -> distilled rule or tiny policy
  -> standalone compute_demand file
  -> export validation
```

The important distinction is:

| Term | Meaning |
|---|---|
| Environment | The executable RL API, such as `MarketGameEnv` or `GymMarketGameEnv`. |
| Scenario | A named, repeatable matchup loaded into the environment. It defines demand profile, seed, and opponent policies. |
| Teacher | A trained RL policy or heuristic used to generate decisions for distillation. |
| Student | The smaller exported policy, usually an interpretable rule or embedded tiny model. |

This keeps training flexible while preserving the final competition constraint:
the submitted house should be a plain Python `compute_demand(price, hour,
battery_charge, demand, price_history)` implementation.

## Why Scenarios

Scenario-driven experiments make comparisons repeatable. Instead of hard-coding
opponents in every script, a scenario gives a name to the whole experimental
setup:

- base demand profile;
- random seed;
- opponent population;
- opponent policy parameters.

The default scenario registry is:

```text
python/market_game_downstream/rl/scenario_configs/weekly.json
```

That file is intentionally small and useful for smoke tests. Larger training
populations live in:

```text
python/market_game_downstream/rl/scenario_configs/large_population.json
```

Held-out validation scenarios live in:

```text
python/market_game_downstream/rl/scenario_configs/held_out_validation.json
```

These are intended for final policy selection and distillation validation, not
for teacher-action collection over the default weekly curriculum.

Invalid-demand diagnostics are exercised by:

```text
python/market_game_downstream/rl/scenario_configs/invalid_demand_stress.json
```

The scenario config format supports repeated opponent blocks with `count`, plus
`$seed` and `$index` placeholders for policy parameters. That makes it possible
to define 25, 40, or 50 opponent houses without manually listing every agent.
For the full format, including stochastic `grab_bag` populations, see
`scenario_config_schema.md`.

The current default scenarios are:

| Scenario | Purpose |
|---|---|
| `week_1_baselines` | Basic profile with stock heuristic opponents. |
| `week_2_new_profile` | Random demand profile with seeded stochastic opponent. |
| `week_3_mixed_population` | Spike profile with mixed/inference/noisy/oscillating opponents. |
| `week_4_chaotic_houses` | Double-spike profile with more volatile opponent behavior. |

The large-population config currently includes:

| Scenario | Opponent count | Purpose |
|---|---:|---|
| `large_profile1_mixed_25` | 25 | Profile1 with a balanced mix of baseline, price-aware, rolling, noisy, and oscillating agents. |
| `large_random_grab_bag_40` | 40 | Random profile with a seeded grab-bag population. |
| `large_spike_grab_bag_50` | 50 | Double-spike stress scenario with a seeded chaotic grab bag. |

The held-out validation config currently includes:

| Scenario | Opponent count | Purpose |
|---|---:|---|
| `validation_profile1_inference_mix` | 4 | Small profile1 validation mix with legal inference and stochastic threshold behavior. |
| `validation_random_grab_bag_24` | 24 | Random-profile held-out grab bag for multi-seed validation. |
| `validation_dspike_volatile_30` | 30 | Double-spike validation population with volatile and oscillating opponents. |

Evaluate fixed policies in a scenario:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios \
  --scenario week_1_baselines \
  --seed 3
```

Evaluate the whole default curriculum:

```bash
python3 -m python.market_game_downstream.rl.evaluate_scenarios --seed 3
```

## PPO Training

PPO training uses `GymMarketGameEnv`, which wraps the dependency-free
`MarketGameEnv` for RLlib/Gymnasium compatibility.

The default smoke run checks only that RLlib can collect complete 24-hour
episodes:

```bash
python3 -m python.market_game_downstream.rl.training.train_rllib \
  --iterations 1 \
  --observation-mode price_history
```

A more useful run trains against a named scenario, saves a checkpoint, and then
evaluates the trained policy against one or more scenarios:

```bash
python3 -m python.market_game_downstream.rl.training.train_rllib \
  --iterations 20 \
  --observation-mode price_history \
  --scenario week_1_baselines \
  --scenario-seed 3 \
  --checkpoint-dir /tmp/market_game_ppo_week1_seed3_20iter \
  --evaluate-scenario week_1_baselines \
  --evaluate-scenario week_2_new_profile \
  --evaluate-scenario week_3_mixed_population \
  --evaluate-scenario week_4_chaotic_houses
```

For serious training, use the large-population config and more than smoke-test
batch sizes. For example:

```bash
python3 -m python.market_game_downstream.rl.training.train_rllib \
  --iterations 500 \
  --observation-mode price_history \
  --scenario-config python/market_game_downstream/rl/scenario_configs/large_population.json \
  --scenario large_random_grab_bag_40 \
  --scenario-seed 11 \
  --train-batch-size 4096 \
  --minibatch-size 256 \
  --num-epochs 10 \
  --lr 0.0003 \
  --checkpoint-dir /tmp/market_game_ppo_large_random_seed11 \
  --evaluate-scenario large_profile1_mixed_25 \
  --evaluate-scenario large_random_grab_bag_40 \
  --evaluate-scenario large_spike_grab_bag_50
```

This is intentionally much larger than the quick examples. A 1-20 iteration run
is useful for plumbing, but it should not be expected to produce a competitive
agent.

The training script uses RLlib's current new API stack for inference through
`RLModule.forward_inference(...)`. It does not depend on the old
`compute_single_action(...)` path.

## Reading PPO Results

The learner reward is negative cost:

```text
episode_return = -total_cost
```

So less-negative return is better, and reported evaluation `total_cost` is the
number to compare with heuristic agents.

One recent 20-iteration run trained on `week_1_baselines`, seed `3`, with
`price_history` observations. It produced:

| Scenario | PPO total cost |
|---|---:|
| `week_1_baselines` | `33.7525` |
| `week_2_new_profile` | `40.8197` |
| `week_3_mixed_population` | `87.7571` |
| `week_4_chaotic_houses` | `112.5600` |

Interpretation: PPO was running end to end and learned something on the
training scenario, but it was not competitive with the stronger hand-written
heuristics. This is expected for a small 20-iteration run and should be treated
as pipeline validation, not a strong agent result.

## Why Distill

The final submission should not depend on Ray, Torch, Gymnasium, or checkpoint
files unless the competition explicitly permits that. Distillation converts a
flexible training-time policy into a simpler deployment-time policy.

Preferred deployment forms:

| Form | Why |
|---|---|
| Interpretable rule | Easy to audit, robust in closed environments, and competition-safe. |
| Embedded tiny model | More expressive, still self-contained if all weights are literal Python constants. |
| External checkpoint | Avoid unless explicitly allowed. |
| Runtime training | Avoid for closed-VM or classroom competition runs. |

## Current Distillers

The distillation tools collect teacher actions over named scenarios and fit one
of three student policies:

| Student | Use |
|---|---|
| `threshold` | Most interpretable, lowest-capacity rule. |
| `tree` | VIPER-style policy extraction into a small decision tree. |
| `matrix` | Small linear softmax model trained with NumPy and exported as literal matrices. |

The threshold distiller fits this rule:

```text
if price <= cheap_price:
    charge
elif price >= expensive_price:
    discharge
elif hour >= late_hour:
    discharge
else:
    neutral
```

Then it renders a standalone source file containing only constants and a
`compute_demand(...)` function.

The tree distiller is the first VIPER-style implementation. It learns a
decision tree from teacher rollouts using weighted CART-style splits. Full VIPER
uses Q-value gaps to prioritize states where choosing the wrong action is most
costly; this implementation has the sample-weight hook but currently uses
uniform weights for PPO checkpoints because the checkpoint path exposes actions
but not calibrated action-value gaps yet.

The matrix distiller trains a linear softmax classifier locally with NumPy:

```text
action = argmax(W @ normalized_observation + b)
```

The exported file does not import NumPy. It stores `W`, `b`, feature means, and
feature scales as literal Python lists and performs inference with loops.

Run checkpoint distillation:

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

Change `--student threshold` to `--student tree --tree-depth 3` for the
decision-tree student, or to `--student matrix --matrix-epochs 800` for the
matrix student.

Validate the generated submission:

```bash
python3 - <<'PY'
from python.market_game_downstream.rl.export import validate_submission_file

print(validate_submission_file("/tmp/market_game_distilled_threshold.py"))
PY
```

## Reading Distillation Results

The distiller reports:

| Field | Meaning |
|---|---|
| `samples` | Number of teacher decisions collected. |
| `matches` | Number of student decisions matching the teacher action. |
| `accuracy` | Action-match rate, not game performance. |
| `cheap_price`, `expensive_price`, `late_hour` | Fitted rule parameters. |
| `validation_cost` | Cost from a default simulator validation run. |

For the recent PPO checkpoint above, threshold distillation produced:

```text
samples=96
matches=91
accuracy=0.9479
cheap_price=-0.9000
expensive_price=0.1000
late_hour=18
validation_cost=37.1633333333
```

That high match rate did not mean the PPO teacher was strong. The fitted
thresholds reveal that the teacher had effectively learned a mostly
neutral/no-charge behavior. In this case, distillation was useful because it
made the teacher's weak behavior obvious and auditable.

The same checkpoint also distilled into:

| Student | Action-match accuracy | Exported behavior |
|---|---:|---|
| `tree` | `0.9896` | Same scenario costs as PPO teacher. |
| `matrix` | `0.9792` | Same scenario costs as PPO teacher. |

This reinforces the same lesson: the students can imitate the current teacher,
but stronger training is needed before distillation will produce a competitive
submission.

## Current Limitations

- PPO training is still a pipeline baseline, not a tuned agent.
- The threshold distiller is interpretable but deliberately low-capacity.
- The tree distiller is VIPER-style but not yet full Q-weighted VIPER.
- The matrix student is linear; it may need a one-hidden-layer student if the
  teacher learns genuinely nonlinear behavior.
- Action-match accuracy measures imitation, not market performance.
- Current checkpoint evaluation uses one deterministic rollout per scenario.
- A stronger training setup should evaluate across more seeds and should include
  a validation scenario set not used for teacher-action collection.

## Next Work

Good next steps are:

1. Add multi-seed scenario sweeps for PPO evaluation.
2. Train for longer and log checkpoint-by-checkpoint validation costs.
3. Add a second distiller for a tiny embedded neural net or small decision tree.
4. Add a held-out scenario config for distillation validation.
5. Compare distilled policies against `PriceAwarePolicy`, `RollingPricePolicy`,
   and `LegalInferencePolicy` as standard baselines.

The working plan for these items is in `rl_tuning_plan.md`.
