"""Training/export contracts for standalone competition submissions."""

from __future__ import annotations

from dataclasses import dataclass

from python.market_game_downstream.rl.agents.observations import ObservationMode
from python.market_game_downstream.rl.export.validators import MAX_SOURCE_BYTES


@dataclass(frozen=True)
class ExportProfile:
    """Training/export contract for standalone competition submissions."""

    observation_mode: ObservationMode = ObservationMode.PRICE_HISTORY
    action_space: str = "discrete_battery_posture"
    fcnet_hiddens: tuple[int, ...] = (8,)
    activation: str = "tanh"
    max_source_bytes: int = MAX_SOURCE_BYTES

    @property
    def hidden_size(self) -> int:
        self.validate()
        return self.fcnet_hiddens[0]

    def model_config(self) -> dict[str, object]:
        self.validate()
        return {
            "fcnet_hiddens": list(self.fcnet_hiddens),
            "fcnet_activation": self.activation,
        }

    def estimate_source_bytes(self) -> int:
        """Conservative source-size estimate for literal embedded weights."""
        if len(self.fcnet_hiddens) != 1 or self.fcnet_hiddens[0] < 1:
            return self.max_source_bytes + 1
        hidden_size = self.fcnet_hiddens[0]
        input_features = 10
        action_count = 3
        parameter_count = (
            hidden_size * input_features
            + hidden_size
            + action_count * hidden_size
            + action_count
        )
        return 4_000 + parameter_count * 18

    def validate(self) -> None:
        """Fail early when a choice is outside the supported exporter surface."""
        if self.observation_mode != ObservationMode.PRICE_HISTORY:
            raise ValueError("exportable RLlib checkpoints require price_history observations")
        if self.action_space != "discrete_battery_posture":
            raise ValueError("exportable RLlib checkpoints require discrete battery posture actions")
        if len(self.fcnet_hiddens) != 1 or self.fcnet_hiddens[0] < 1:
            raise ValueError("exportable RLlib checkpoints require one positive hidden layer")
        if self.activation != "tanh":
            raise ValueError("exportable RLlib checkpoints require tanh activation")
        if self.max_source_bytes > MAX_SOURCE_BYTES:
            raise ValueError(
                f"export profile max_source_bytes cannot exceed validator limit {MAX_SOURCE_BYTES}"
            )
        estimated_source_bytes = self.estimate_source_bytes()
        if estimated_source_bytes > self.max_source_bytes:
            raise ValueError(
                "exportable RLlib checkpoint is expected to exceed the source-size "
                f"limit: estimate={estimated_source_bytes} limit={self.max_source_bytes}"
            )


EXPORTABLE_PPO_PROFILE = ExportProfile()
