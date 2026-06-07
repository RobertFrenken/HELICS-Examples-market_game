"""Dependency-free Gymnasium-style environment for the market game."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.config import DEFAULT_CONFIG, MarketGameConfig
from ..agents.observations import (
    InferenceBelief,
    InferenceFeatures,
    ObservationContext,
    ObservationMode,
    build_observation,
    update_inference_belief,
)
from ..agents.policies import FollowDemandPolicy
from ..core.rules import (
    BatteryAction,
    action_to_market_load,
)
from ..core.simulator import BatteryState, HouseHourInput, HousePolicy, HourRecord, step_market_hour


@dataclass
class EnvStepDiagnostics:
    own_market_load: float
    own_cost: float
    own_battery: float
    total_market_load: float
    average_market_load: float
    next_price: float
    opponent_loads: dict[str, float] = field(default_factory=dict)


class MarketGameEnv:
    """Single-agent RL environment embedded in a multi-house market.

    The learning agent controls one house through a discrete battery action.
    Opponents are ordinary ``HousePolicy`` instances. Observations are built
    only from legal local inputs, own action history, and delayed price history.
    Hidden aggregate values are exposed only through ``info`` diagnostics.
    """

    def __init__(
        self,
        opponent_policies: list[HousePolicy] | None = None,
        config: MarketGameConfig = DEFAULT_CONFIG,
        observation_mode: ObservationMode | str = ObservationMode.PRICE_HISTORY,
        final_battery_target: float | None = None,
        final_battery_penalty: float = 0.0,
    ):
        self.config = config
        self.opponent_policies = opponent_policies or [FollowDemandPolicy(), FollowDemandPolicy()]
        self.observation_mode = ObservationMode(observation_mode)
        self.final_battery_target = final_battery_target
        self.final_battery_penalty = final_battery_penalty

        self.hour = 0
        self.current_price = config.initial_price
        self.price_history: list[float] = []
        self.own_market_load_history: list[float] = []
        self.own_cost_history: list[float] = []
        self.records: list[HourRecord] = []
        self.own_battery = BatteryState(config.initial_battery, config)
        self.opponent_batteries: list[BatteryState] = []
        self.inference_belief = InferenceBelief()
        self.last_inference_features = InferenceFeatures()

    @property
    def house_count(self) -> int:
        return 1 + len(self.opponent_policies)

    @property
    def demand(self) -> list[float]:
        return self.config.demand_profile

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, object]]:
        del seed
        self.hour = 0
        self.current_price = self.config.initial_price
        self.price_history = [self.current_price]
        self.own_market_load_history = []
        self.own_cost_history = []
        self.records = []
        self.own_battery = BatteryState(self.config.initial_battery, self.config)
        self.opponent_batteries = [
            BatteryState(self.config.initial_battery, self.config)
            for _ in self.opponent_policies
        ]
        self.inference_belief.reset()
        self.last_inference_features = InferenceFeatures()
        for policy in self.opponent_policies:
            reset = getattr(policy, "reset", None)
            if reset is not None:
                reset()

        return self._make_observation(), self._make_info()

    def step(self, action: BatteryAction | int) -> tuple[list[float], float, bool, bool, dict[str, object]]:
        if self.hour >= self.config.episode_hours:
            raise RuntimeError("episode is already terminated; call reset()")

        base_demand = self.demand[self.hour]
        own_proposed_load = action_to_market_load(
            action,
            base_demand,
            self.own_battery.energy,
            self.config,
        )
        hour_inputs = [
            HouseHourInput(
                name="learner",
                proposed_market_load=own_proposed_load,
                base_demand=base_demand,
                battery=self.own_battery,
            )
        ]
        hour_inputs.extend(self._opponent_hour_inputs(base_demand))
        record, house_results = step_market_hour(
            hour=self.hour,
            price=self.current_price,
            house_inputs=hour_inputs,
            config=self.config,
        )
        own_result = house_results[0]
        opponent_loads = {
            result.name: result.market_load for result in house_results[1:]
        }

        own_market_load = own_result.market_load
        own_cost = own_result.cost
        self.own_market_load_history.append(own_market_load)
        self.own_cost_history.append(own_cost)

        self.records.append(record)

        reward = -own_cost
        self.current_price = record.next_price
        self.hour += 1
        terminated = self.hour >= self.config.episode_hours
        truncated = False
        if not terminated:
            self.price_history.append(self.current_price)
            self._update_inference_features()
        reward = self._apply_terminal_penalty(reward, terminated)

        info = self._make_info(
            EnvStepDiagnostics(
                own_market_load=own_market_load,
                own_cost=own_cost,
                own_battery=self.own_battery.energy,
                total_market_load=record.total_load,
                average_market_load=record.average_load,
                next_price=record.next_price,
                opponent_loads=opponent_loads,
            )
        )
        return self._make_observation(), reward, terminated, truncated, info

    def _opponent_hour_inputs(self, base_demand: float) -> list[HouseHourInput]:
        hour_inputs: list[HouseHourInput] = []
        for policy, battery in zip(self.opponent_policies, self.opponent_batteries):
            proposed = policy.compute_demand(
                self.current_price,
                self.hour,
                battery.energy,
                self.demand,
                self.price_history,
            )
            hour_inputs.append(
                HouseHourInput(
                    name=policy.name,
                    proposed_market_load=proposed,
                    base_demand=base_demand,
                    battery=battery,
                )
            )
        return hour_inputs

    def _apply_terminal_penalty(self, reward: float, terminated: bool) -> float:
        if not terminated:
            return reward
        if self.final_battery_target is None or not self.final_battery_penalty:
            return reward
        return reward - self.final_battery_penalty * abs(
            self.own_battery.energy - self.final_battery_target
        )

    def _update_inference_features(self) -> None:
        if self.observation_mode != ObservationMode.INFERENCE:
            return
        self.last_inference_features = update_inference_belief(
            self._make_context(),
            self.inference_belief,
        )

    def _make_context(self) -> ObservationContext:
        hour = min(self.hour, self.config.episode_hours - 1)
        return ObservationContext(
            hour=hour,
            price=self.current_price,
            battery_charge=self.own_battery.energy,
            demand=self.demand,
            price_history=self.price_history,
            own_market_load_history=self.own_market_load_history,
            house_count=self.house_count,
            config=self.config,
        )

    def _make_observation(self) -> list[float]:
        return build_observation(
            self.observation_mode,
            self._make_context(),
            self.last_inference_features,
        )

    def _make_info(self, diagnostics: EnvStepDiagnostics | None = None) -> dict[str, object]:
        info: dict[str, object] = {
            "hour": self.hour,
            "price": self.current_price,
            "battery": self.own_battery.energy,
            "total_cost": sum(self.own_cost_history),
        }
        if diagnostics is not None:
            info["diagnostics"] = diagnostics
        return info
