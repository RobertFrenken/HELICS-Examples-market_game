"""Distill trained RL actions into self-contained market-game submissions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

from ..core.rules import BatteryAction, action_to_market_load
from ..envs.env import MarketGameEnv
from ..agents.observations import ObservationMode
from ..scenarios import CompetitionScenario, load_scenarios, scenario_by_name


TeacherActionFn = Callable[[Any], int]

ACTION_INDEX_TO_BATTERY_ACTION = {
    0: BatteryAction.DISCHARGE,
    1: BatteryAction.NEUTRAL,
    2: BatteryAction.CHARGE,
}
BATTERY_ACTION_TO_ACTION_INDEX = {
    BatteryAction.DISCHARGE: 0,
    BatteryAction.NEUTRAL: 1,
    BatteryAction.CHARGE: 2,
}


@dataclass(frozen=True)
class TeacherSample:
    """One legal observation plus the teacher's discrete battery action."""

    scenario: str
    observation: list[float]
    hour: int
    price: float
    battery_charge: float
    base_demand: float
    action: BatteryAction


@dataclass(frozen=True)
class ThresholdRule:
    """Interpretable battery-action rule fitted to teacher decisions."""

    cheap_price: float
    expensive_price: float
    late_hour: int = 21

    def action(self, price: float, hour: int, battery_charge: float) -> BatteryAction:
        if price <= self.cheap_price:
            return BatteryAction.CHARGE
        if price >= self.expensive_price and battery_charge > 0.0:
            return BatteryAction.DISCHARGE
        if hour >= self.late_hour and battery_charge > 0.0:
            return BatteryAction.DISCHARGE
        return BatteryAction.NEUTRAL


@dataclass(frozen=True)
class DistillationReport:
    """Fit quality for a distilled threshold rule."""

    rule: object
    samples: int
    matches: int

    @property
    def accuracy(self) -> float:
        if self.samples == 0:
            return 0.0
        return self.matches / self.samples


@dataclass(frozen=True)
class TreeNode:
    """Small classification tree node for extracted battery-action policies."""

    prediction: BatteryAction
    feature: int | None = None
    threshold: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


@dataclass(frozen=True)
class DecisionTreePolicy:
    """Interpretable decision-tree student policy."""

    root: TreeNode
    observation_mode: ObservationMode

    def action(self, observation: list[float]) -> BatteryAction:
        node = self.root
        while not node.is_leaf:
            if node.left is None or node.right is None or node.feature is None:
                break
            node = node.left if observation[node.feature] <= node.threshold else node.right
        return node.prediction


@dataclass(frozen=True)
class MatrixPolicy:
    """Tiny linear softmax student trained as a NumPy matrix locally."""

    weights: list[list[float]]
    bias: list[float]
    feature_mean: list[float]
    feature_scale: list[float]
    observation_mode: ObservationMode

    def action(self, observation: list[float]) -> BatteryAction:
        normalized = [
            (value - mean) / scale
            for value, mean, scale in zip(observation, self.feature_mean, self.feature_scale)
        ]
        scores = [
            bias + sum(weight * value for weight, value in zip(row, normalized))
            for row, bias in zip(self.weights, self.bias)
        ]
        return ACTION_INDEX_TO_BATTERY_ACTION[max(range(len(scores)), key=scores.__getitem__)]


def collect_teacher_samples(
    teacher_action: TeacherActionFn,
    scenario_names: list[str],
    observation_mode: str,
    scenario_config: str | None = None,
    scenario_seed: int | None = None,
) -> list[TeacherSample]:
    """Roll out a trained teacher and record its legal battery decisions."""
    scenarios = _load_named_scenarios(scenario_names, scenario_config, scenario_seed)
    samples: list[TeacherSample] = []
    for scenario in scenarios:
        opponent_policies, market_config = scenario.to_env_config()
        env = MarketGameEnv(
            opponent_policies=opponent_policies,
            config=market_config,
            observation_mode=observation_mode,
        )
        obs, info = env.reset(seed=scenario.seed)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            hour = int(info["hour"])
            price = float(info["price"])
            battery = float(info["battery"])
            action_index = int(teacher_action(obs))
            battery_action = ACTION_INDEX_TO_BATTERY_ACTION[action_index]
            samples.append(
                TeacherSample(
                    scenario=scenario.name,
                    observation=[float(value) for value in obs],
                    hour=hour,
                    price=price,
                    battery_charge=battery,
                    base_demand=env.demand[hour],
                    action=battery_action,
                )
            )
            obs, _reward, terminated, truncated, info = env.step(battery_action)
    return samples


def fit_threshold_rule(samples: list[TeacherSample]) -> DistillationReport:
    """Fit the best small threshold rule to teacher battery actions."""
    if not samples:
        raise ValueError("cannot distill from zero teacher samples")

    candidate_prices = sorted({round(sample.price, 4) for sample in samples})
    cheap_candidates = [min(candidate_prices) - 1.0, *candidate_prices]
    expensive_candidates = [*candidate_prices, max(candidate_prices) + 1.0]
    late_hour_candidates = [18, 19, 20, 21, 22, 23, 24]

    best_rule = ThresholdRule(cheap_price=0.12, expensive_price=0.49)
    best_matches = -1
    for cheap_price in cheap_candidates:
        for expensive_price in expensive_candidates:
            if cheap_price > expensive_price:
                continue
            for late_hour in late_hour_candidates:
                rule = ThresholdRule(
                    cheap_price=cheap_price,
                    expensive_price=expensive_price,
                    late_hour=late_hour,
                )
                matches = sum(
                    1
                    for sample in samples
                    if rule.action(
                        sample.price,
                        sample.hour,
                        sample.battery_charge,
                    )
                    == sample.action
                )
                if matches > best_matches:
                    best_rule = rule
                    best_matches = matches

    return DistillationReport(
        rule=best_rule,
        samples=len(samples),
        matches=best_matches,
    )


def fit_decision_tree_policy(
    samples: list[TeacherSample],
    observation_mode: str,
    max_depth: int = 3,
    min_leaf: int = 2,
    sample_weights: list[float] | None = None,
) -> DistillationReport:
    """Fit a small VIPER-style decision tree to teacher actions.

    This is the policy-extraction part of VIPER: the tree imitates an oracle
    policy on states collected from rollouts. Full VIPER also prioritizes
    samples by teacher Q-value gaps; callers can approximate that behavior by
    passing ``sample_weights``.
    """
    if not samples:
        raise ValueError("cannot distill from zero teacher samples")
    weights = sample_weights or [1.0] * len(samples)
    if len(weights) != len(samples):
        raise ValueError("sample_weights must match samples length")

    indices = list(range(len(samples)))
    root = _build_tree(samples, weights, indices, depth=0, max_depth=max_depth, min_leaf=min_leaf)
    policy = DecisionTreePolicy(root=root, observation_mode=ObservationMode(observation_mode))
    matches = sum(1 for sample in samples if policy.action(sample.observation) == sample.action)
    return DistillationReport(rule=policy, samples=len(samples), matches=matches)


def fit_matrix_policy(
    samples: list[TeacherSample],
    observation_mode: str,
    epochs: int = 800,
    learning_rate: float = 0.1,
    l2: float = 0.001,
) -> DistillationReport:
    """Fit a tiny linear softmax policy with NumPy and literal weight export."""
    if not samples:
        raise ValueError("cannot distill from zero teacher samples")
    import numpy as np

    x = np.asarray([sample.observation for sample in samples], dtype=np.float64)
    y = np.asarray(
        [BATTERY_ACTION_TO_ACTION_INDEX[sample.action] for sample in samples],
        dtype=np.int64,
    )
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale

    class_count = 3
    weights = np.zeros((class_count, x.shape[1]), dtype=np.float64)
    bias = np.zeros(class_count, dtype=np.float64)
    target = np.eye(class_count, dtype=np.float64)[y]

    for _ in range(epochs):
        logits = x @ weights.T + bias
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        error = (probs - target) / len(samples)
        weights -= learning_rate * (error.T @ x + l2 * weights)
        bias -= learning_rate * error.sum(axis=0)

    policy = MatrixPolicy(
        weights=weights.tolist(),
        bias=bias.tolist(),
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        observation_mode=ObservationMode(observation_mode),
    )
    matches = sum(1 for sample in samples if policy.action(sample.observation) == sample.action)
    return DistillationReport(rule=policy, samples=len(samples), matches=matches)


def _build_tree(
    samples: list[TeacherSample],
    weights: list[float],
    indices: list[int],
    depth: int,
    max_depth: int,
    min_leaf: int,
) -> TreeNode:
    prediction = _majority_action(samples, weights, indices)
    if depth >= max_depth or len(indices) <= 2 * min_leaf or _is_pure(samples, indices):
        return TreeNode(prediction=prediction)

    split = _best_split(samples, weights, indices, min_leaf)
    if split is None:
        return TreeNode(prediction=prediction)

    feature, threshold, left_indices, right_indices = split
    return TreeNode(
        prediction=prediction,
        feature=feature,
        threshold=threshold,
        left=_build_tree(samples, weights, left_indices, depth + 1, max_depth, min_leaf),
        right=_build_tree(samples, weights, right_indices, depth + 1, max_depth, min_leaf),
    )


def _best_split(
    samples: list[TeacherSample],
    weights: list[float],
    indices: list[int],
    min_leaf: int,
) -> tuple[int, float, list[int], list[int]] | None:
    feature_count = len(samples[0].observation)
    parent_impurity = _weighted_gini(samples, weights, indices)
    best_gain = 0.0
    best_split = None
    for feature in range(feature_count):
        values = sorted({samples[index].observation[feature] for index in indices})
        thresholds = [(left + right) / 2.0 for left, right in zip(values, values[1:])]
        for threshold in thresholds:
            left_indices = [
                index for index in indices if samples[index].observation[feature] <= threshold
            ]
            left_set = set(left_indices)
            right_indices = [index for index in indices if index not in left_set]
            if len(left_indices) < min_leaf or len(right_indices) < min_leaf:
                continue
            left_weight = sum(weights[index] for index in left_indices)
            right_weight = sum(weights[index] for index in right_indices)
            total_weight = left_weight + right_weight
            impurity = (
                left_weight / total_weight * _weighted_gini(samples, weights, left_indices)
                + right_weight / total_weight * _weighted_gini(samples, weights, right_indices)
            )
            gain = parent_impurity - impurity
            if gain > best_gain:
                best_gain = gain
                best_split = (feature, threshold, left_indices, right_indices)
    return best_split


def _weighted_gini(
    samples: list[TeacherSample],
    weights: list[float],
    indices: list[int],
) -> float:
    total = sum(weights[index] for index in indices)
    if total <= 0.0:
        return 0.0
    impurity = 1.0
    for action in BatteryAction:
        mass = sum(weights[index] for index in indices if samples[index].action == action)
        probability = mass / total
        impurity -= probability * probability
    return impurity


def _majority_action(
    samples: list[TeacherSample],
    weights: list[float],
    indices: list[int],
) -> BatteryAction:
    masses = {
        action: sum(weights[index] for index in indices if samples[index].action == action)
        for action in BatteryAction
    }
    return max(masses, key=masses.__getitem__)


def _is_pure(samples: list[TeacherSample], indices: list[int]) -> bool:
    return len({samples[index].action for index in indices}) == 1


def render_threshold_submission(rule: ThresholdRule) -> str:
    """Render a standalone ``compute_demand`` source file for a threshold rule."""
    return f'''"""Distilled market-game threshold policy."""

CHEAP_PRICE = {rule.cheap_price!r}
EXPENSIVE_PRICE = {rule.expensive_price!r}
LATE_HOUR = {rule.late_hour!r}
BATTERY_CAPACITY = 20.0
MAX_CHARGE = 5.0
MAX_DISCHARGE = 10.0


def compute_demand(price, hour, battery_charge, demand, price_history):
    """Return the desired market load for one house and one hour."""
    del price_history
    base_demand = demand[hour]
    remaining_capacity = BATTERY_CAPACITY - battery_charge

    if price <= CHEAP_PRICE and remaining_capacity > 0.0:
        return base_demand + min(MAX_CHARGE, remaining_capacity)

    if price >= EXPENSIVE_PRICE and battery_charge > 0.0:
        return base_demand - min(MAX_DISCHARGE, battery_charge)

    if hour >= LATE_HOUR and battery_charge > 0.0:
        return base_demand - min(MAX_DISCHARGE, battery_charge)

    return base_demand
'''


def render_tree_submission(policy: DecisionTreePolicy) -> str:
    """Render a standalone ``compute_demand`` source file for a tree policy."""
    return _render_policy_submission(
        observation_mode=policy.observation_mode,
        model_constants=f"TREE = {_tree_to_literal(policy.root)!r}\n",
        predict_function='''def _predict_action(features):
    node = TREE
    while node[0] == "node":
        _kind, feature, threshold, left, right = node
        node = left if features[feature] <= threshold else right
    return node[1]
''',
    )


def render_matrix_submission(policy: MatrixPolicy) -> str:
    """Render a standalone ``compute_demand`` source file for a matrix policy."""
    return _render_policy_submission(
        observation_mode=policy.observation_mode,
        model_constants=(
            f"FEATURE_MEAN = {policy.feature_mean!r}\n"
            f"FEATURE_SCALE = {policy.feature_scale!r}\n"
            f"WEIGHTS = {policy.weights!r}\n"
            f"BIAS = {policy.bias!r}\n"
        ),
        predict_function='''def _predict_action(features):
    normalized = [
        (value - mean) / scale
        for value, mean, scale in zip(features, FEATURE_MEAN, FEATURE_SCALE)
    ]
    scores = []
    for row, bias in zip(WEIGHTS, BIAS):
        score = bias
        for weight, value in zip(row, normalized):
            score += weight * value
        scores.append(score)
    best = 0
    for index in range(1, len(scores)):
        if scores[index] > scores[best]:
            best = index
    return best
''',
    )


def _render_policy_submission(
    observation_mode: ObservationMode,
    model_constants: str,
    predict_function: str,
) -> str:
    return f'''"""Distilled market-game policy."""

import math

BATTERY_CAPACITY = 20.0
MAX_CHARGE = 5.0
MAX_DISCHARGE = 10.0
{model_constants}

{_render_feature_function(observation_mode)}

{predict_function}

def _market_load(action, base_demand, battery_charge):
    if action == 0:
        return base_demand - min(MAX_DISCHARGE, battery_charge)
    if action == 2:
        return base_demand + min(MAX_CHARGE, BATTERY_CAPACITY - battery_charge)
    return base_demand


def compute_demand(price, hour, battery_charge, demand, price_history):
    """Return the desired market load for one house and one hour."""
    base_demand = demand[hour]
    features = _features(price, hour, battery_charge, demand, price_history)
    action = _predict_action(features)
    return _market_load(action, base_demand, battery_charge)
'''


def _render_feature_function(observation_mode: ObservationMode) -> str:
    if observation_mode == ObservationMode.LOCAL:
        return '''def _features(price, hour, battery_charge, demand, price_history):
    del price_history
    angle = 2.0 * math.pi * hour / 24.0
    return [
        math.sin(angle),
        math.cos(angle),
        price,
        battery_charge,
        demand[hour],
    ]
'''
    if observation_mode == ObservationMode.PRICE_HISTORY:
        return '''def _features(price, hour, battery_charge, demand, price_history):
    angle = 2.0 * math.pi * hour / 24.0
    previous_prices = price_history[:-1]
    if len(previous_prices) >= 1:
        price_lag_1 = previous_prices[-1]
    else:
        price_lag_1 = price
    if len(previous_prices) >= 2:
        price_lag_2 = previous_prices[-2]
        price_trend = previous_prices[-1] - previous_prices[-2]
    else:
        price_lag_2 = price
        price_trend = 0.0
    recent = previous_prices[-6:]
    if recent:
        price_mean = sum(recent) / len(recent)
        price_volatility = (
            sum((value - price_mean) ** 2 for value in recent) / len(recent)
        ) ** 0.5
    else:
        price_mean = price
        price_volatility = 0.0
    return [
        math.sin(angle),
        math.cos(angle),
        price,
        battery_charge,
        demand[hour],
        price_lag_1,
        price_lag_2,
        price_mean,
        price_trend,
        price_volatility,
    ]
'''
    raise ValueError("standalone distillation currently supports local and price_history observations")


def _tree_to_literal(node: TreeNode) -> tuple:
    if node.is_leaf:
        return ("leaf", BATTERY_ACTION_TO_ACTION_INDEX[node.prediction])
    if node.left is None or node.right is None or node.feature is None:
        return ("leaf", BATTERY_ACTION_TO_ACTION_INDEX[node.prediction])
    return (
        "node",
        node.feature,
        node.threshold,
        _tree_to_literal(node.left),
        _tree_to_literal(node.right),
    )


def write_threshold_submission(rule: ThresholdRule, path: str | Path) -> Path:
    """Write a standalone threshold-rule submission file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_threshold_submission(rule), encoding="utf-8")
    return output_path


def write_tree_submission(policy: DecisionTreePolicy, path: str | Path) -> Path:
    """Write a standalone decision-tree submission file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_tree_submission(policy), encoding="utf-8")
    return output_path


def write_matrix_submission(policy: MatrixPolicy, path: str | Path) -> Path:
    """Write a standalone matrix-policy submission file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_matrix_submission(policy), encoding="utf-8")
    return output_path


def action_for_threshold_rule(
    rule: ThresholdRule,
    base_demand: float,
    price: float,
    hour: int,
    battery_charge: float,
) -> float:
    """Return the legal market load implied by a fitted threshold rule."""
    return action_to_market_load(
        rule.action(price, hour, battery_charge),
        base_demand,
        battery_charge,
    )


def load_teacher_from_checkpoint(
    checkpoint_path: str | Path,
    observation_mode: str = ObservationMode.PRICE_HISTORY.value,
) -> tuple[object, TeacherActionFn]:
    """Load a trained RLlib algorithm checkpoint as a teacher action function."""
    from ray.tune.registry import register_env

    from ..training.train_rllib import ENV_NAME, _compute_action, build_ppo_config, make_env

    register_env(ENV_NAME, make_env)
    algorithm = build_ppo_config(observation_mode=observation_mode).build_algo()
    algorithm.restore(Path(checkpoint_path).as_posix())
    return algorithm, lambda obs: _compute_action(algorithm, obs)


def load_teacher_from_module(path: str | Path, function_name: str = "teacher_action") -> TeacherActionFn:
    """Load a lightweight teacher action function from a Python module."""
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location("_market_game_teacher", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load teacher module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    teacher_action = getattr(module, function_name, None)
    if not callable(teacher_action):
        raise ValueError(f"teacher module must define callable {function_name}")
    return teacher_action


def _load_named_scenarios(
    scenario_names: list[str],
    scenario_config: str | None,
    scenario_seed: int | None,
) -> list[CompetitionScenario]:
    scenarios = load_scenarios(scenario_config, seed=scenario_seed)
    return [scenario_by_name(name, scenarios) for name in scenario_names]
