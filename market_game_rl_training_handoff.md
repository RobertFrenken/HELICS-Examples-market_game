# HELICS Market Game: RL Model + Training Handoff

This file is intended to pair with `market_game_house_observability.md`.

The observability file defines what a legal house agent can and cannot infer. This file defines how to turn that framing into an RL training setup and a contest-safe deployment strategy for the HELICS-Examples `market_game` branch.

---

## 1. Core Objective

Build a house-agent policy that replaces or augments:

```python
compute_demand(price, hour, battery_charge, demand, price_history)
```

The policy should choose an hourly market purchase/demand value that minimizes the house's total energy cost over the episode while obeying the legal observation constraints.

The deployed agent must behave as a legal house:

- It may use only the arguments passed into `compute_demand`.
- It may store its own past actions/state in module-level or function-level persistent variables if the game runner allows it.
- It may compute its own cumulative cost.
- It may use the known battery rules and pricing rules.
- It may infer delayed aggregate market behavior from price history.
- It may not observe other individual houses' actions, batteries, costs, code, or identities.

The intended RL setup is therefore a **single-agent partially observable RL problem embedded in a multi-agent market simulation**.

---

## 2. Deployment Constraint

Assume the closed VM will only run the submitted Python demand function file.

Therefore, avoid relying on:

- external `.pt`, `.pkl`, `.onnx`, `.joblib`, or checkpoint files;
- Stable-Baselines imports at runtime;
- PyTorch imports at runtime;
- TensorFlow imports at runtime;
- unknown local file paths;
- network access;
- package versions not guaranteed by the closed VM.

Preferred deployment modes:

| Mode | Description | Safety | RL purity |
|---|---|---:|---:|
| Distilled heuristic | Train RL offline, then manually compress behavior into rules | Very high | Low-medium |
| Embedded tiny policy | Hard-code learned model weights as Python lists/constants | High | Medium-high |
| External model file | Load model weights from submitted artifact | Low unless explicitly allowed | High |
| Runtime RL training | Learn during competition run | Low | Medium |

Recommended final submission strategy:

```text
Train RL offline
  -> evaluate and interpret policy
  -> distill into compact rule logic OR tiny embedded neural net
  -> deploy as a self-contained compute_demand implementation
```

---

## 3. Environment Abstraction

Create a pure Python simulator before trying to train through HELICS.

Reason:

- RL needs thousands to millions of episodes.
- HELICS co-simulation is valuable for final integration, but too slow and cumbersome for early training.
- A pure Python simulator makes it easier to test timing, price lag, battery constraints, and reward accounting.

Target architecture:

```text
HELICS market_game code
        |
        v
Extract market rules into pure Python simulator
        |
        v
Wrap simulator as Gymnasium-style environment
        |
        v
Train RL policy offline
        |
        v
Export distilled/embedded policy
        |
        v
Drop into compute_demand(...)
```

The pure simulator should preserve the original market timing:

- At hour `t`, each house chooses market demand.
- The market maker aggregates submitted demand.
- The resulting aggregate demand determines the next price.
- Therefore, price at hour `t` reflects aggregate market demand from hour `t-1`.

Do not accidentally leak same-hour aggregate information into the RL observation.

---

## 4. Agent Observation Design

The policy observation should include only legal features.

### Tier 0: Local-Only Observation

Use this for the first baseline RL agent.

```python
obs = [
    hour_sin,
    hour_cos,
    current_price,
    own_battery_charge,
    own_base_demand_now,
]
```

Notes:

- Encode hour cyclically.
- Normalize price, battery, and demand.
- This agent sees no explicit market reconstruction.

### Tier 1: Add Price History Features

Use this for the main non-inference RL baseline.

```python
obs = [
    hour_sin,
    hour_cos,
    current_price,
    own_battery_charge,
    own_base_demand_now,
    price_lag_1,
    price_lag_2,
    price_lag_3,
    price_mean_recent,
    price_trend_recent,
    price_volatility_recent,
]
```

Useful derived features:

```python
price_lag_1 = price_history[-1] if len(price_history) >= 1 else current_price
price_lag_2 = price_history[-2] if len(price_history) >= 2 else current_price
price_mean_recent = mean(price_history[-k:]) if price_history else current_price
price_trend_recent = price_history[-1] - price_history[-2] if len(price_history) >= 2 else 0.0
price_volatility_recent = std(price_history[-k:]) if len(price_history) >= k else 0.0
```

### Tier 2: Add Legal Aggregate Inference Features

This is the main research-grade observation set.

It uses the observability bedrock:

```python
avg_market_purchase_lag_1 = invert_price_to_avg_purchase(price_t)
others_avg_purchase_lag_1 = (
    (N * avg_market_purchase_lag_1 - own_purchase_lag_1) / (N - 1)
)
crowd_net_charge_lag_1 = others_avg_purchase_lag_1 - shared_base_demand_lag_1
estimated_crowd_battery = update_belief_state(crowd_net_charge_lag_1)
```

Possible Tier 2 observation:

```python
obs = [
    hour_sin,
    hour_cos,
    current_price,
    own_battery_charge,
    own_base_demand_now,

    price_lag_1,
    price_lag_2,
    price_mean_recent,
    price_trend_recent,
    price_volatility_recent,

    inferred_avg_market_purchase_lag_1,
    inferred_others_avg_purchase_lag_1,
    inferred_crowd_net_charge_lag_1,
    estimated_crowd_battery_mean,
    tier_boundary_risk,
    anomaly_score,
]
```

Important: hidden simulator variables may be used for training labels, diagnostics, and evaluation, but not as policy inputs.

---

## 5. Action Space

Prefer a simple discrete action space for the MVP.

### Recommended: Battery Action

The policy chooses battery posture rather than raw demand.

```python
action in {-1, 0, +1}
```

| Action | Meaning | Market purchase |
|---|---|---|
| `-1` | discharge / buy less than base demand | `base_demand - discharge_amount` |
| `0` | neutral / follow base demand | `base_demand` |
| `+1` | charge / buy more than base demand | `base_demand + charge_amount` |

Then clamp to feasible battery and demand constraints.

Example conversion:

```python
def action_to_purchase(action, base_demand, battery_charge, params):
    if action == -1:
        desired_delta = -params.max_discharge_rate
    elif action == 0:
        desired_delta = 0.0
    elif action == +1:
        desired_delta = params.max_charge_rate
    else:
        raise ValueError("invalid action")

    feasible_delta = clamp_delta_to_battery_limits(
        desired_delta,
        battery_charge,
        params.battery_capacity,
        params.max_charge_rate,
        params.max_discharge_rate,
    )

    purchase = base_demand + feasible_delta
    return max(0.0, purchase)
```

### Alternative: Multi-Level Discrete Action

For more expressiveness:

```python
action_delta in [-max_discharge, -0.5*max_discharge, 0, 0.5*max_charge, max_charge]
```

This is still easy to deploy and much easier than a continuous action model.

### Avoid Initially: Raw Continuous Demand

Raw continuous demand is flexible, but it makes constraint enforcement and training harder.

---

## 6. Battery Constraint Handling

The deployed policy should never return impossible values if the game expects legal demand.

Enforce constraints after policy output.

Generic logic:

```python
def clamp_purchase(base, proposed_purchase, battery_charge, params):
    # delta > 0 means charging battery
    # delta < 0 means discharging battery
    delta = proposed_purchase - base

    max_charge_possible = min(
        params.max_charge_rate,
        params.battery_capacity - battery_charge,
    )

    max_discharge_possible = min(
        params.max_discharge_rate,
        battery_charge,
        base,  # avoid negative market purchase
    )

    delta = min(delta, max_charge_possible)
    delta = max(delta, -max_discharge_possible)

    return max(0.0, base + delta)
```

If exact battery semantics differ in the branch, Codex should inspect and match the existing implementation.

---

## 7. Reward Function

Primary objective:

```python
reward_t = - price_t * own_market_purchase_t
```

Episode objective:

```python
episode_reward = -total_cost
```

Recommended augmented training reward:

```python
episode_reward = (
    - total_cost
    - lambda_final_battery * abs(final_battery - target_final_battery)
    - lambda_violation * constraint_violation_count
)
```

Why include final battery penalty?

Without it, the RL agent may drain the battery at the end of the 24-hour episode and appear artificially good. A final state-of-charge penalty makes the policy less myopic and closer to repeated-day behavior.

Possible target:

```python
target_final_battery = initial_battery
```

or

```python
target_final_battery = 0.5 * battery_capacity
```

depending on game rules.

---

## 8. Gymnasium-Style Simulator Sketch

Codex should implement this only after inspecting the actual market game files.

The target shape is:

```python
class MarketGameEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.hour = 0
        self.price_history = []
        self.own_action_history = []
        self.own_purchase_history = []
        self.own_battery = self.config.initial_battery

        self.other_houses = make_other_house_policies(self.config)
        self.other_batteries = initialize_other_batteries(self.config)

        self.current_price = self.config.initial_price

        obs = self._make_observation()
        info = {}
        return obs, info

    def step(self, action):
        base = self.demand_profile[self.hour]

        own_purchase = self._action_to_purchase(action, base, self.own_battery)

        other_purchases = []
        for house in self.other_houses:
            purchase = house.compute_demand(
                price=self.current_price,
                hour=self.hour,
                battery_charge=house.battery_charge,
                demand=self.demand_profile,
                price_history=self.price_history,
            )
            purchase = clamp_purchase(...)
            other_purchases.append(purchase)

        avg_purchase = np.mean([own_purchase] + other_purchases)

        next_price = pricing_rule(avg_purchase)

        reward = -self.current_price * own_purchase

        self._update_battery_states(own_purchase, other_purchases)
        self.own_purchase_history.append(own_purchase)
        self.price_history.append(self.current_price)

        self.current_price = next_price
        self.hour += 1

        terminated = self.hour >= 24
        truncated = False

        if terminated:
            reward -= self._final_battery_penalty()

        obs = self._make_observation()
        info = {
            "own_purchase": own_purchase,
            "avg_purchase": avg_purchase,
            "next_price": next_price,
            "total_cost": self.total_cost,
        }

        return obs, reward, terminated, truncated, info
```

Main point: keep price lag consistent with the original market maker.

---

## 9. Training Algorithms

Start with one of:

| Algorithm | Use when | Notes |
|---|---|---|
| DQN | discrete actions, simple observation | Easy baseline |
| PPO discrete | discrete actions, robust default | Recommended first serious attempt |
| PPO continuous | continuous action | Later only |
| SAC / TD3 | continuous action, larger training budget | Probably overkill initially |

Recommended first serious setup:

```text
Algorithm: PPO
Action space: Discrete(3)
Episode length: 24 hours
Observation: Tier 1, then Tier 2
Opponents: fixed honest heuristics first
Reward: negative cost + final battery penalty
```

For a Codex implementation, use Stable-Baselines3 during training if available. Do not assume it exists on the closed VM.

---

## 10. Opponent Policy Curriculum

Training against a single fixed opponent behavior can overfit badly.

Use staged curricula:

| Phase | Other houses | Purpose |
|---|---|---|
| 1 | Fixed honest baseline | Learn basic arbitrage and constraints |
| 2 | Honest threshold policies with varied parameters | Avoid brittle timing |
| 3 | Noisy honest policies | Handle stochastic aggregate behavior |
| 4 | Mixed policy population | Generalize across house strategies |
| 5 | Include volatility-seeking adversaries | Robustness |
| 6 | Randomize number/fraction of adversaries | Robust deployment |

Example honest opponent types:

```python
class FollowDemandHouse:
    # Always returns base demand.
    pass

class ThresholdBatteryHouse:
    # Charges below low price threshold and discharges above high threshold.
    pass

class RollingMeanHouse:
    # Compares current price to recent moving average.
    pass

class ConservativeHouse:
    # Preserves battery for projected future high-price periods.
    pass
```

Example adversary types:

```python
class RandomFeasibleHouse:
    # Random feasible charge/neutral/discharge.
    pass

class SpikeAmplifierHouse:
    # Charges during high aggregate demand periods to worsen price spikes.
    pass

class VolatilityHouse:
    # Alternates aggressive charge/discharge to induce price movement.
    pass

class AntiPriceHouse:
    # Behaves opposite of cost-minimizing intuition.
    pass
```

The RL learner should not receive opponent labels. It only sees legal aggregate signals.

---

## 11. Domain Randomization

To avoid overfitting to a single exact game configuration, randomize training episodes.

Potential randomized values:

```python
episode_config = {
    "num_houses": sample([5, 10, 20]),
    "initial_price": sample_range(...),
    "initial_battery": sample_range(...),
    "battery_capacity": fixed_or_sampled,
    "charge_rate": fixed_or_sampled,
    "demand_profile_variant": sample(...),
    "opponent_policy_mix": sample(...),
    "malicious_fraction": sample([0.0, 0.1, 0.2]),
}
```

Do not randomize facts that are fixed by the competition unless using the randomized setup specifically for robustness testing.

---

## 12. Evaluation Plan

Evaluate at least these agents:

| Agent | Purpose |
|---|---|
| Follow-demand baseline | No battery strategy |
| Greedy threshold heuristic | Simple price arbitrage |
| Rolling mean heuristic | Simple adaptive strategy |
| Legal inference heuristic | Uses price inversion and crowd battery belief |
| RL local-only | Tests RL without aggregate inference |
| RL price-history | Tests price dynamics learning |
| RL belief-state | Tests value of observability/inference features |
| Oracle upper bound | Illegal, for evaluation only |

Important: the oracle must never be used as a deployable policy. It is only a performance ceiling.

Metrics:

| Metric | Meaning |
|---|---|
| Total cost | Primary objective |
| Regret vs oracle | How far from upper bound |
| Cost vs heuristic | Whether RL is worthwhile |
| Final battery charge | Detects end-of-episode cheating |
| Constraint violations | Should be zero |
| Price volatility | Market-level side effect |
| Robustness degradation | How performance changes under adversaries |
| Inference error | Quality of aggregate reconstruction |
| Tier crossing count | Whether policy avoids price cliffs |

Suggested evaluation matrix:

```text
Rows: agent type
Columns: opponent population scenario
Cells: mean cost, std cost, final battery, violations, volatility
```

Opponent scenarios:

```text
1. all honest follow-demand
2. all honest threshold
3. mixed honest
4. mixed honest + noisy
5. 10% adversarial
6. 20% adversarial
7. worst-case volatility adversary
```

---

## 13. Legal Inference Feature Implementation

This section depends on the actual pricing rule.

Codex should inspect the market maker code to identify:

- price function;
- demand aggregation logic;
- exact lag timing;
- price bounds;
- flat regions or thresholds;
- number of houses;
- battery capacity and rate rules.

Then implement:

```python
def invert_price_to_avg_purchase(price):
    """
    Return an estimated previous-hour average market purchase.

    If exact inversion is impossible because the price rule is piecewise,
    flat, clipped, or lossy, return the midpoint of the feasible inverse
    interval and optionally expose uncertainty.
    """
    ...
```

For piecewise or clipped pricing, prefer returning:

```python
estimated_avg_purchase
inverse_low
inverse_high
inverse_width
```

Useful features:

```python
inversion_uncertainty = inverse_high - inverse_low
tier_boundary_risk = distance_to_nearest_pricing_threshold(estimated_avg_purchase)
```

Others-only estimate:

```python
def estimate_others_avg_purchase(avg_market_purchase, own_purchase, n_houses):
    if n_houses <= 1:
        return avg_market_purchase
    return (n_houses * avg_market_purchase - own_purchase) / (n_houses - 1)
```

Crowd net battery movement:

```python
crowd_net_charge = others_avg_purchase - shared_base_demand_for_that_hour
```

Crowd battery belief:

```python
crowd_battery_belief = clamp(
    previous_belief + crowd_net_charge,
    0.0,
    battery_capacity,
)
```

Caveat: this is a belief over average crowd state, not a known fact.

---

## 14. Self-Contained Deployment: Embedded Tiny Neural Policy

If using a true learned policy in the closed VM, export a tiny network as literal Python constants.

Example deployment shape:

```python
# Learned offline; pasted into submission file.
W1 = [...]
B1 = [...]
W2 = [...]
B2 = [...]

def relu(x):
    return x if x > 0.0 else 0.0

def tiny_policy_forward(features):
    hidden = []
    for i in range(len(W1)):
        z = B1[i]
        for j in range(len(features)):
            z += W1[i][j] * features[j]
        hidden.append(relu(z))

    logits = []
    for i in range(len(W2)):
        z = B2[i]
        for j in range(len(hidden)):
            z += W2[i][j] * hidden[j]
        logits.append(z)

    return argmax(logits)
```

Action mapping:

```python
action_index = tiny_policy_forward(features)

if action_index == 0:
    proposed_purchase = base_demand - discharge_amount
elif action_index == 1:
    proposed_purchase = base_demand
else:
    proposed_purchase = base_demand + charge_amount

return clamp_purchase(base_demand, proposed_purchase, battery_charge, params)
```

Recommended model size:

```text
Input dimension: 8-16 features
Hidden layer: 8-32 units
Output: 3 actions
Activation: ReLU or tanh
No dependencies beyond Python math
```

Benefits:

- no model file needed;
- no PyTorch needed;
- deterministic;
- easy to inspect;
- likely compatible with closed VM;
- still represents a trained RL policy.

---

## 15. Self-Contained Deployment: Distilled Heuristic Policy

A robust heuristic may outperform a poorly trained RL model.

Template:

```python
def compute_demand(price, hour, battery_charge, demand, price_history):
    base = demand[hour]

    # History features
    recent = price_history[-6:] if price_history else [price]
    avg_recent = sum(recent) / len(recent)
    trend = 0.0
    if len(price_history) >= 2:
        trend = price_history[-1] - price_history[-2]

    # Lookahead demand pressure from shared demand profile
    future_window = [demand[(hour + i) % 24] for i in range(1, 4)]
    future_pressure = sum(future_window) / len(future_window)

    # Simple price regime
    cheap = price < 0.95 * avg_recent
    expensive = price > 1.05 * avg_recent

    # Defensive behavior
    if expensive and battery_charge > 0.2:
        proposed = base - 1.0
    elif cheap and battery_charge < 0.8:
        proposed = base + 1.0
    else:
        proposed = base

    return clamp_purchase(base, proposed, battery_charge, params)
```

Possible improvements:

- use price volatility to become more conservative;
- use inferred crowd battery belief to save energy before expected spikes;
- avoid charging near tier thresholds;
- reduce aggressiveness if own action likely worsens next price;
- preserve final battery as episode approaches hour 23.

---

## 16. State Persistence Inside `compute_demand`

If allowed, use module-level globals to store own history.

Example:

```python
_STATE = {
    "last_seen_episode": None,
    "own_purchase_history": [],
    "price_history_seen": [],
    "estimated_crowd_battery": None,
    "cumulative_cost": 0.0,
}
```

Reset logic is important. The function may be called across multiple episodes/runs.

Possible reset detection:

```python
if hour == 0:
    reset_state()
```

or, if price history is cleared:

```python
if len(price_history) == 0:
    reset_state()
```

Store after computing demand:

```python
_STATE["own_purchase_history"].append(returned_purchase)
_STATE["price_history_seen"].append(price)
_STATE["cumulative_cost"] += price * returned_purchase
```

Be careful with the one-hour lag:

- the own action at hour `t-1` pairs with price observed at hour `t` when reconstructing aggregate demand;
- current action at hour `t` affects future price, not current price.

---

## 17. Common Failure Modes

| Failure | Consequence | Prevention |
|---|---|---|
| Training with illegal hidden states | Policy will not deploy legally | Strict observation wrapper |
| Ignoring price lag | Learns wrong causality | Align `price[t]` with demand at `t-1` |
| No final battery penalty | End-of-day battery draining | Add terminal state penalty |
| External model file | Closed VM fails | Embed weights or distill |
| Package dependency | Import error | Pure Python deployment |
| Overfitting to one opponent | Brittle strategy | Opponent curriculum |
| Unclamped actions | Invalid demand or battery state | Always enforce constraints |
| Miscounting number of houses | Bad others-only inference | Read config or use fallback |
| Assuming individual attribution | Illegal/false reasoning | Stay aggregate-level |
| Training only on adversaries | Poor normal performance | Mix honest and adversarial populations |

---

## 18. Recommended Codex Task Breakdown

Use this file as the high-level spec. Codex should proceed in small patches.

### Task 1: Inspect Existing Market Game

Find and summarize:

- `compute_demand` signature and call site;
- market maker pricing rule;
- aggregation formula;
- battery update rule;
- battery capacity/rate constants;
- number of houses / config source;
- episode length;
- whether `price_history` includes current price or only prior prices;
- whether submitted demand is clamped elsewhere.

### Task 2: Extract Pure Simulator

Create a pure Python module, for example:

```text
market_game_rl/
  simulator.py
  policies.py
  features.py
  env.py
  train.py
  evaluate.py
```

Keep it independent of HELICS.

### Task 3: Implement Baseline Policies

Implement:

- follow-demand;
- threshold;
- rolling-mean;
- legal-inference heuristic.

### Task 4: Implement Gymnasium Environment

Build `MarketGameEnv`.

Verify:

- reset works;
- one full 24-hour episode runs;
- battery never violates constraints;
- price lag is correct;
- reward equals negative cost;
- info dict logs hidden simulator variables for diagnostics only.

### Task 5: Train PPO/DQN

Train at least:

- local-only observation;
- price-history observation;
- belief-state observation.

Save training artifacts locally for development only.

### Task 6: Evaluate

Generate a comparison table:

```text
agent, opponent_scenario, mean_cost, std_cost, final_battery, violations, price_volatility
```

### Task 7: Export Deployment Policy

Either:

1. generate `compute_demand_distilled.py`, or
2. generate `compute_demand_embedded_nn.py`.

Deployment file must be self-contained.

### Task 8: Clean Submission

Final file should:

- avoid external dependencies;
- avoid reading files;
- define only needed helper functions/constants;
- clamp outputs;
- reset internal state at hour 0;
- be deterministic unless randomness is explicitly allowed and seeded.

---

## 19. Minimal Acceptance Criteria

A successful MVP should satisfy:

```text
[ ] Pure Python simulator can run 1000+ episodes quickly.
[ ] Gymnasium environment exposes legal observations only.
[ ] At least two baselines are implemented.
[ ] RL policy beats follow-demand baseline in honest setting.
[ ] RL policy does not violate battery constraints.
[ ] Evaluation includes adversarial/noisy populations.
[ ] Deployment policy is self-contained inside compute_demand file.
[ ] No external model files are required at runtime.
[ ] Price lag is correctly handled.
[ ] Observability constraints from companion file are respected.
```

---

## 20. One-Sentence Project Framing

Train a legal-observation house policy that uses local battery state, delayed price signals, and aggregate market reconstruction to minimize energy cost under honest and adversarial market populations, then deploy the learned behavior as a self-contained `compute_demand(...)` implementation compatible with a closed VM.
