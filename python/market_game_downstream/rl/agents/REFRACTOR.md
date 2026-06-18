# Agent Composition Refactor

This note is the implementation reference for refactoring `rl/agents` toward
small, declarative, model-agnostic building blocks. The goal is to support
hand-written baselines, tunable algorithms, search-optimized strategies, and
neural/RL policies through the same agent composition path.

## Design Goal

An agent should be modeled first as a decision-making system:

```text
percept history + state/belief -> action
```

In this market game, the deployed simulator/export ABI still requires a
market-load function:

```python
compute_demand(price, hour, battery_charge, demand, price_history) -> float
```

The refactor should preserve that ABI by making composed agents implement
`compute_demand(...)`. Internally, the agent should use a domain-level
percept/action/controller grammar:

```text
MarketPercept -> Controller -> MarketAction -> market load
```

Reinforcement learning is one possible controller implementation. It should not
define the package structure or the core interfaces.

## Sourced Primitive Model

The stronger cross-reference across AI agents, reinforcement learning,
multi-agent environments, and control-oriented libraries is:

- `Environment`: the outside system that evolves when actions are applied.
- `Agent`: the entity being evaluated; it receives percepts and emits actions.
- `Percept` / `Observation`: information available to the agent at a point in
  time. A percept is the raw legal input; an observation is often an encoded
  representation for an algorithm.
- `State` / `Belief` / `Memory`: agent-owned information carried across time.
- `Controller` / `Policy` / `AgentFunction`: maps percept history and state to
  actions.
- `Action`: the semantic choice made by the agent.
- `Reward` / `Objective` / `PerformanceMeasure`: evaluates behavior.
- `Experience` / `Transition`: optional record used by learning rules.
- `Learner` / `Optimizer`: optional component that updates controller
  parameters from experience.
- `WorldModel`: optional model of environment dynamics, useful for planning or
  model-based control.

Design justification:

- Russell and Norvig's AIMA frames an agent as something that perceives its
  environment through sensors and acts through actuators; the key abstraction
  is the agent function mapping percept sequences to actions. This means the
  controller over percept history is more fundamental than a feature extractor.
  Source: <https://aima.cs.berkeley.edu/4th-ed/pdfs/newchap02.pdf>
- Sutton and Barto's RL framing centers the loop on agent, environment, state
  or observation, action, reward, policy, value function, and optional model.
  Feature extraction is an implementation detail of some policies, not a core
  agent primitive. Source:
  <https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf>
- Gymnasium defines the interoperability boundary at the environment step:
  action in, observation/reward/termination/info out. It standardizes the
  environment-agent exchange, not agent internals. Source:
  <https://gymnasium.farama.org/api/env/>
- PettingZoo extends the same exchange to multi-agent systems with agents,
  observations, actions, rewards, terminations, truncations, and infos. Source:
  <https://pettingzoo.farama.org/api/aec/> and
  <https://pettingzoo.farama.org/api/parallel/>
- Stable-Baselines3 exposes feature extractors as policy-network internals.
  That supports treating feature extraction as a subcomponent of learned/vector
  controllers rather than as a peer primitive for every agent. Source:
  <https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html>
- scikit-learn separates constructor/tunable parameters from learned state via
  `get_params` and `set_params`. That supports optional tunability without
  requiring all agents to be explicitly learnable. Source:
  <https://scikit-learn.org/stable/developers/develop.html>

## Amended Target Primitives

Use these as the durable public grammar:

```python
class Controller(Protocol):
    def decide(self, percept: MarketPercept, state: AgentState) -> MarketAction:
        ...


class AgentState(Protocol):
    def reset(self) -> None:
        ...


class Learner(Protocol):
    def update(self, experience: Experience) -> None:
        ...


class FeatureExtractor(Protocol):
    def encode(self, percept: MarketPercept, state: AgentState) -> Observation:
        ...
```

`FeatureExtractor` is intentionally lower-level than `Controller`. Hardcoded
controllers should be able to decide directly from `MarketPercept` without
being forced through a numeric observation vector. Learned/vector controllers
can compose a feature extractor internally.

Use semantic market actions instead of raw floats where possible:

```python
class FollowDemand:
    ...


class BatteryDelta:
    kwh: float


class TargetLoad:
    load: float


class BatteryPostureAction:
    posture: BatteryPosture
```

The market-load projection is still needed, but it is an implementation
boundary from semantic action to simulator ABI, not the primary concept.

## Current Shape

The current `agents` package already contains useful pieces:

- `observations.py`: legal feature construction for vector-based controllers.
- `actions.py`: semantic market actions emitted by controllers.
- `action_spaces.py`: learner action spaces used by RL/Gym adapters.
- `policies.py`: simulator-facing `compute_demand(...)` adapters over
  controllers.
- `callables.py`: adapters around the official `compute_demand(...)` shape.

The remaining ownership boundary is learned inference. Learned/vector
controllers should own their feature extractor and action decoder internally;
training algorithms, replay buffers, value functions, and reward shaping stay
outside `agents/`.

## Optional Capabilities

Do not force every strategy to be learnable or trainable. Model those as
optional capabilities.

```python
class Tunable(Protocol):
    def get_params(self) -> dict[str, object]:
        ...

    def set_params(self, **params: object) -> None:
        ...


class Trainable(Protocol):
    def update(self, transition: Transition) -> None:
        ...
```

Examples:

- `FollowDemandController`: implements only `Controller`.
- `RollingThresholdController`: implements `Controller` and `Tunable`.
- `TorchPolicyController`: implements `Controller`, `Tunable`, and maybe
  `Trainable`.
- `RLlibController`: likely wraps inference only; RLlib owns training.

This keeps "learnable parameters" agnostic. They may be thresholds, reserves,
lookup-table entries, linear weights, neural-network weights, or external
checkpoint state.

## Composer

The central user-facing object should wire primitives together and implement
the official policy ABI.

```python
@dataclass
class MarketAgent:
    name: str
    controller: Controller
    action_projector: ActionProjector
    state: AgentState = field(default_factory=NoAgentState)

    def reset(self) -> None:
        self.state.reset()
        reset_if_supported(self.controller)
        reset_if_supported(self.action_projector)

    def compute_demand(
        self,
        price: float,
        hour: int,
        battery_charge: float,
        demand: list[float],
        price_history: list[float],
    ) -> float:
        percept = MarketPercept(
            price=price,
            hour=hour,
            battery_charge=battery_charge,
            demand=demand,
            price_history=price_history,
        )
        action = self.controller.decide(percept, self.state)
        return self.action_projector.market_load(action, percept)
```

The composer should remain small. It should not know whether the controller is
hardcoded, tuned, optimized, or learned.

## Current Directory Layout

```text
agents/
  __init__.py
  percepts.py         # MarketPercept
  actions.py          # semantic MarketAction classes
  action_spaces.py    # learner action spaces for RL/Gym adapters
  experiences.py      # Experience/Transition records
  state.py            # NoAgentState, DictAgentState, inference beliefs
  interfaces.py       # Controller, ActionProjector, AgentState, Tunable, Trainable
  compose.py          # MarketAgent
  features.py         # legal domain feature helpers
  observations.py     # vector feature builders for learned controllers
  projectors.py       # semantic market action -> proposed market load
  registry.py         # declarative construction from dict/TOML
  controllers/
    __init__.py
    baselines.py     # follow demand, flatten demand, full cycle
    thresholds.py    # price-aware, rolling-price, noisy-threshold
    inference.py     # legal inference and belief-driven controllers
    learned.py       # wrappers around learned/vector controller implementations
```

## Current Migration Status

The package is now structurally aligned with the target grammar:

```text
MarketPercept + AgentState -> Controller -> MarketAction -> MarketActionProjector -> market load
```

Completed:

- `MarketPercept` lives in `percepts.py`; `MarketContext` and `contexts.py` are
  removed.
- Semantic market actions live in `actions.py`; `market_actions.py` is removed.
- `MarketAgent` is controller-only. The old `Observer -> Strategy -> Actuator`
  composition path is removed.
- `strategies/`, `observers.py`, and `actuators.py` are removed.
- Baseline, threshold, noisy, inference, oscillating, invalid-demand, and
  volatility-seeking decision logic lives under `controllers/`.
- `policies.py` is a simulator-facing `compute_demand(...)` adapter layer over
  `MarketAgent` and controller instances.
- Learned/vector inference has a controller surface in `controllers/learned.py`:
  feature extractor + vector model + action decoder -> semantic `MarketAction`.
- Feature extractors in `observations.py` encode directly from `MarketPercept`
  and `AgentState`; there is no separate observation context value object.
- Inference memory lives in `state.py` as `InferenceBeliefState` and is passed
  through `MarketAgent.state`.
- RL/Gym learner action spaces in `action_spaces.py` decode learner outputs to
  semantic market actions. Projection to market load is owned by
  `MarketActionProjector`.
- `registry.py` builds controllers, nested feature extractors, nested action
  decoders, action projectors, and state objects from dictionaries.

Known stale surfaces after the structural migration:

- Some tests and smoke checks still assume learner action spaces expose
  `market_load(...)`; they need to be updated to decode actions and project
  through `MarketActionProjector`.
- Export code still renders a standalone policy function directly instead of
  sharing the new `TinyTanhController`/decoder path.
- Training code still deals primarily in Gym/RLlib environment actions rather
  than explicitly exporting or evaluating controller objects.

Remaining architecture work:

1. Update export/runtime inference so checkpoint-derived policies use or mirror
   `TinyTanhController`, `PriceHistoryFeatureExtractor`, and
   `BatteryPostureIndexDecoder`.
2. Make evaluation/training boundaries explicit about whether they are handling
   a controller, a `MarketAgent`, or the standalone `compute_demand(...)` ABI.
3. Update tests and smoke checks to the new action-space decode/projector split.
4. Decide whether `policies.py` remains as the long-term simulator ABI adapter
   or whether scenario construction should move directly to declarative
   `MarketAgent` configs.

## Target End State

The final package should make the durable domain concepts obvious and push
training-library details to adapter modules.

```text
agents/
  __init__.py
  percepts.py          # MarketPercept
  actions.py           # semantic MarketAction classes
  controllers/
    __init__.py
    baselines.py       # follow demand, flatten demand, full cycle
    thresholds.py      # price-aware, rolling-price, noisy-threshold
    inference.py       # legal inference and belief-driven controllers
    learned.py         # wrappers around learned controller implementations
  state.py             # NoAgentState, DictAgentState, inference beliefs
  projectors.py        # MarketAction -> proposed market load
  compose.py           # MarketAgent
  features.py          # reusable legal feature helpers
  observations.py      # vector encodings for learned/RL controllers
  registry.py          # declarative construction
```

Consolidation targets:

- `policies.py` exposes simulator-facing `compute_demand(...)` adapters over
  controllers; controller logic lives under `controllers/`.
- `strategies/` is removed; useful classes moved to `controllers/`.
- `observers.py` is removed; vector encodings stay in `observations.py`.
- `actuators.py` is removed; semantic actions live in `actions.py`,
  projection lives in `projectors.py`, and RL/Gym learner actions live in
  `action_spaces.py`.
- `MarketContext` is removed; callers use `MarketPercept`.
- `Observer`, `Strategy`, and `Actuator` are no longer public primitives.

## Where RL Fits

An RL policy is not a new top-level primitive. It is a particular way to
implement or train a controller. In a clean decomposition, RL spans multiple
optional pieces:

```text
MarketAgent
  controller = LearnedController / RLController
    feature_extractor: MarketPercept -> Observation
    model: Observation -> encoded action/logits/value/etc.
    action_decoder: model output -> MarketAction
  action_projector = MarketActionProjector
  state = InferenceBeliefState / NoAgentState
  learner = PPO/DQN/etc. training algorithm, optional and usually outside runtime
  objective = reward function used during training
```

Runtime inference path:

```text
MarketPercept
  -> feature extractor
  -> vector observation
  -> learned model / policy network
  -> action decoder
  -> semantic MarketAction
  -> MarketActionProjector
  -> proposed market load
```

Training path:

```text
Experience or vector Transition
  -> learner/optimizer
  -> updated controller parameters
```

This means:

- A hardcoded controller can ignore feature extraction entirely and decide
  directly from `MarketPercept`.
- A threshold controller may expose tunable parameters with `get_params` and
  `set_params`, but it does not need a learner.
- An RL controller contains a feature extractor, vector model, and action
  decoder as internal parts of the controller.
- The RL algorithm itself usually belongs in `training/`, not in `agents/`.
  `agents/` should only expose the trained or trainable controller interface.
- A value function, replay buffer, optimizer, exploration schedule, and reward
  shaping are training concerns. They may be needed to learn a controller, but
  they are not required for a deployed `compute_demand(...)` house.

## Declarative Construction

Support agent construction from plain dictionaries so TOML/YAML/JSON can be
used without coupling the core to one config format.

Example TOML shape:

```toml
[agent]
name = "rolling_price"

[agent.controller]
type = "rolling_threshold"
cheap_ratio = 0.94
expensive_ratio = 1.08
reserve = 4.0

[agent.action_projector]
type = "market_action"

[agent.state]
type = "inference_belief"
```

Equivalent Python:

```python
agent = MarketAgent(
    name="rolling_price",
    controller=RollingThresholdController(
        cheap_ratio=0.94,
        expensive_ratio=1.08,
        reserve=4.0,
    ),
    action_projector=MarketActionProjector(),
)
```

## Migration Plan

1. Add `MarketPercept`, semantic `MarketAction` classes, `Controller`, and
   `ActionProjector` without changing existing behavior.
2. Migrate preset `compute_demand` policies to controller-backed wrappers,
   starting with the simplest baselines.
3. Keep `observations.py` as vector feature construction for learned/vector
   controllers, not as a mandatory step for every agent.
4. Keep learner action spaces for Gym/RL integration under `action_spaces.py`.
5. Add registry support for declarative controller/projector construction.
6. Update environment/training code to depend on composed agents where useful,
   while preserving the simulator-facing `HousePolicy` and `compute_demand(...)`
   ABI.

## External Design Inspiration

- Gymnasium: keep environment `reset`/`step` separate from agent internals.
  Environments exchange observations, actions, rewards, and info rather than
  model-specific concepts.
- RLlib: learned and heuristic policies can coexist and be routed separately.
  This supports keeping RL-specific adapters outside the core agent composer.
- PettingZoo: useful model for simultaneous multi-agent systems with separate
  per-agent observations and actions.
- scikit-learn: use explicit constructor parameters plus `get_params` and
  `set_params` for tunable hyperparameters, separate from learned fitted state.

## Design Rule

Keep the durable abstraction centered on the market-game decision:

```text
legal percept + state/belief -> controller -> semantic market action -> market load
```

Everything else is an optional capability layered on top.
