"""Learned/vector controller adapters for market-game agents."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from ..primitives import (
    BatteryDelta,
    BatteryPosture,
    MarketAction,
    TargetLoad,
)
from ...observations import PriceHistoryFeatureExtractor
from ..primitives import MarketPercept

Observation = list[float]


class VectorModel(Protocol):
    """Map a vector observation to a vector action representation."""

    def predict(self, observation: Observation) -> object:
        """Return model output for one observation."""


class ActionDecoder(Protocol):
    """Decode model output into a semantic market action."""

    def decode(self, output: object, percept: MarketPercept) -> MarketAction:
        """Return a semantic action for model output."""


@dataclass(frozen=True)
class BatteryPostureIndexDecoder:
    """Decode a 3-action index into a semantic battery posture action."""

    def decode(self, output: object, percept: MarketPercept) -> BatteryPosture:
        del percept
        index = int(_scalar_or_argmax(output))
        if index == 0:
            return BatteryPosture.DISCHARGE
        if index == 1:
            return BatteryPosture.NEUTRAL
        if index == 2:
            return BatteryPosture.CHARGE
        raise ValueError(f"battery posture index must be 0, 1, or 2; got {index!r}")


@dataclass(frozen=True)
class BatteryDeltaDecoder:
    """Decode model output as a signed battery delta in kWh."""

    def decode(self, output: object, percept: MarketPercept) -> BatteryDelta:
        del percept
        return BatteryDelta(float(_scalar_or_argmax(output)))


@dataclass(frozen=True)
class NormalizedBatteryDeltaDecoder:
    """Decode output in [-1, 1] into the configured legal battery-delta range."""

    def decode(self, output: object, percept: MarketPercept) -> BatteryDelta:
        value = max(-1.0, min(1.0, float(_scalar_or_argmax(output))))
        if value >= 0.0:
            return BatteryDelta(value * percept.config.max_charge)
        return BatteryDelta(value * percept.config.max_discharge)


@dataclass(frozen=True)
class TargetLoadDecoder:
    """Decode model output as an explicit target market load."""

    def decode(self, output: object, percept: MarketPercept) -> TargetLoad:
        del percept
        return TargetLoad(float(_scalar_or_argmax(output)))


@dataclass
class VectorController:
    """object that owns feature extraction, model inference, and decoding."""

    feature_extractor: object
    model: VectorModel
    action_decoder: ActionDecoder

    def reset(self) -> None:
        reset = getattr(self.feature_extractor, "reset", None)
        if reset is not None:
            reset()
        reset = getattr(self.model, "reset", None)
        if reset is not None:
            reset()
        reset = getattr(self.action_decoder, "reset", None)
        if reset is not None:
            reset()

    def decide(self, percept: MarketPercept, state: object) -> MarketAction:
        observation = self.feature_extractor.encode(percept, state)
        output = self.model.predict(observation)
        return self.action_decoder.decode(output, percept)


@dataclass(frozen=True)
class TinyTanhModel:
    """One-hidden-layer tanh model for small exported/checkpoint-derived actors."""

    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]

    def predict(self, observation: Observation) -> list[float]:
        hidden = []
        for row, bias in zip(self.w1, self.b1):
            hidden.append(math.tanh(_dot(row, observation) + bias))
        return [_dot(row, hidden) + bias for row, bias in zip(self.w2, self.b2)]


@dataclass
class TinyTanhController(VectorController):
    """Feature-extractor-backed tiny tanh controller with semantic decoding."""

    def __init__(
        self,
        w1: list[list[float]],
        b1: list[float],
        w2: list[list[float]],
        b2: list[float],
        feature_extractor: object | None = None,
        action_decoder: ActionDecoder | None = None,
    ) -> None:
        super().__init__(
            feature_extractor=feature_extractor or PriceHistoryFeatureExtractor(),
            model=TinyTanhModel(w1=w1, b1=b1, w2=w2, b2=b2),
            action_decoder=action_decoder or BatteryPostureIndexDecoder(),
        )


def _dot(row: list[float], values: list[float]) -> float:
    total = 0.0
    for index, weight in enumerate(row):
        total += weight * values[index]
    return total


def _scalar_or_argmax(value: object) -> float | int:
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return float(value[0])
        best = 0
        for index, candidate in enumerate(value[1:], start=1):
            if candidate > value[best]:
                best = index
        return best
    return float(value)
