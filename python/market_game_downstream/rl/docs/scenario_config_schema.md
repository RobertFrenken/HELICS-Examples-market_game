# Scenario Config Schema

Scenario configs are JSON files that define repeatable training and evaluation
matchups. They are meant to be easy to author by hand: choose a demand profile,
choose a seed, then define the opponent population either exactly or as a
stochastic grab bag.

The default examples are:

```text
python/market_game_downstream/rl/scenario_configs/weekly.json
python/market_game_downstream/rl/scenario_configs/large_population.json
```

## Top Level

```json
{
  "seed": 1,
  "scenarios": [
    {
      "name": "my_training_scenario",
      "profile_type": "random",
      "opponents": []
    }
  ]
}
```

Fields:

| Field | Required | Meaning |
|---|---:|---|
| `seed` | No | File-level default seed. |
| `scenarios` | Yes | List of named scenario objects. |

Scenario fields:

| Field | Required | Meaning |
|---|---:|---|
| `name` | Yes | CLI name used by `--scenario`. |
| `profile_type` | No | Demand profile name, such as `profile1`, `random`, `spike`, `dspike`, or `profile_solar`. |
| `seed` | No | Per-scenario seed. Overridden by CLI `--seed` or `--scenario-seed`. |
| `opponents` | Yes | List of explicit agents, repeated blocks, or grab-bag blocks. |

## Available Agent Types

Use these strings in `type` fields or directly in `opponents`:

```text
FlattenDemandPolicy
FollowDemandPolicy
FullCyclePolicy
LegalInferencePolicy
NoisyThresholdPolicy
OscillatingPolicy
PriceAwarePolicy
RollingPricePolicy
VolatilitySeekingPolicy
```

Each type accepts the constructor kwargs from
`python/market_game_downstream/rl/agents/policies.py`. Common kwargs include:

| Policy | Useful kwargs |
|---|---|
| `NoisyThresholdPolicy` | `name`, `seed`, `reserve`, `noise_scale` |
| `OscillatingPolicy` | `name`, `period`, `phase` |
| `RollingPricePolicy` | `name`, `window`, `cheap_ratio`, `expensive_ratio`, `reserve` |
| `LegalInferencePolicy` | `name`, `house_count` |

## Explicit Agents

Use a string for a single default-constructed agent:

```json
{
  "opponents": [
    "PriceAwarePolicy",
    "RollingPricePolicy"
  ]
}
```

Use an object when you need kwargs:

```json
{
  "opponents": [
    {
      "type": "NoisyThresholdPolicy",
      "kwargs": {
        "name": "Noisy_0",
        "seed": "$seed",
        "noise_scale": 0.06
      }
    }
  ]
}
```

## Repeated Blocks

Use `count` to create many agents of the same type:

```json
{
  "type": "PriceAwarePolicy",
  "count": 10,
  "kwargs": {
    "name": "PriceAware_$local_index"
  }
}
```

This creates `PriceAware_0` through `PriceAware_9`.

## Grab-Bag Populations

Use `grab_bag` when you want a stochastic population sampled from available
agent types:

```json
{
  "count": 40,
  "seed": "$seed",
  "name_template": "$type_$index",
  "grab_bag": [
    {
      "type": "PriceAwarePolicy",
      "weight": 2
    },
    {
      "type": "RollingPricePolicy",
      "weight": 2
    },
    {
      "type": "NoisyThresholdPolicy",
      "weight": 3,
      "kwargs": {
        "seed": "$seed+$index",
        "noise_scale": 0.06
      }
    },
    {
      "type": "LegalInferencePolicy",
      "weight": 1,
      "kwargs": {
        "house_count": "$house_count"
      }
    }
  ]
}
```

The `weight` values control relative sampling frequency. The result is
deterministic for a fixed scenario seed.

`name_template` is applied to choices that do not already provide `kwargs.name`.
Use it to avoid duplicate policy names in simulator diagnostics.

## Placeholders

Placeholders work in `kwargs`, `seed`, and `name_template` values.

| Placeholder | Meaning |
|---|---|
| `$seed` | Scenario seed after CLI override. |
| `$index` | Global opponent index within the scenario. |
| `$local_index` | Index within the repeated block or grab-bag block. |
| `$house_count` | Total houses including the learner. |
| `$type` | Policy type without the `Policy` suffix. |
| `$seed+$index` | Integer seed plus global index. |
| `$seed-$index` | Integer seed minus global index. |
| `$seed+$local_index` | Integer seed plus local index. |
| `$seed-$local_index` | Integer seed minus local index. |

String placeholders can also be embedded:

```json
{
  "name": "$type_$index"
}
```

## Complete Training Scenario Example

```json
{
  "seed": 11,
  "scenarios": [
    {
      "name": "custom_large_random_100",
      "profile_type": "random",
      "opponents": [
        {
          "type": "PriceAwarePolicy",
          "count": 20,
          "kwargs": {
            "name": "PriceAware_$local_index"
          }
        },
        {
          "count": 80,
          "seed": "$seed+100",
          "name_template": "$type_$index",
          "grab_bag": [
            {
              "type": "RollingPricePolicy",
              "weight": 2
            },
            {
              "type": "NoisyThresholdPolicy",
              "weight": 4,
              "kwargs": {
                "seed": "$seed+$index",
                "noise_scale": 0.08
              }
            },
            {
              "type": "VolatilitySeekingPolicy",
              "weight": 1
            },
            {
              "type": "LegalInferencePolicy",
              "weight": 1,
              "kwargs": {
                "house_count": "$house_count"
              }
            }
          ]
        }
      ]
    }
  ]
}
```

Run it with:

```bash
python3 -m python.market_game_downstream.rl.training.train_rllib \
  --scenario-config path/to/custom_scenarios.json \
  --scenario custom_large_random_100 \
  --scenario-seed 11 \
  --iterations 500 \
  --train-batch-size 4096 \
  --minibatch-size 256 \
  --num-epochs 10
```

## Design Notes

- Keep `weekly.json` small for fast checks.
- Put serious training populations in a separate config.
- Prefer unique names for generated agents.
- Use `$house_count` for inference policies so the config stays correct when
  the population size changes.
- Treat grab-bag scenarios as stochastic but reproducible: change the scenario
  seed to get a new population draw.
