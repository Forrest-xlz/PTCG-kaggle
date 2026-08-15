"""Pretrained encoder backbone and scalar value model."""
from __future__ import annotations

from collections.abc import Mapping

import torch

from model.network import ModelConfig, PTCGTransformer


GLOBAL_TOKEN_INDEX = 25

_POLICY_ONLY_ATTRIBUTES = (
    "action_input_norm",
    "option_type_embedding",
    "option_context_embedding",
    "option_candidate_embedding",
    "option_target_embedding",
    "option_attack_embedding",
    "option_number_embedding",
    "option_count_embedding",
    "option_player_relation_embedding",
    "option_area_embedding",
    "option_in_play_area_embedding",
    "option_special_condition_embedding",
    "option_numeric_projection",
    "pokemon_dynamic_projection",
    "attack_dynamic_projection",
    "option_token_mlp",
    "attack_feature_projection",
    "no_action_embedding",
    "decoder",
    "decoder_fc",
)


class PTCGEncoderBackbone(PTCGTransformer):
    """The policy network with every current-action/decoder module removed."""

    def __init__(
        self,
        config: ModelConfig,
        card_feature_table: torch.Tensor,
        attack_feature_table: torch.Tensor,
    ) -> None:
        super().__init__(config, card_feature_table, attack_feature_table)
        for name in _POLICY_ONLY_ATTRIBUTES:
            delattr(self, name)

    @classmethod
    def from_policy_state(
        cls,
        config: ModelConfig,
        card_feature_table: torch.Tensor,
        attack_feature_table: torch.Tensor,
        state_dict: Mapping[str, torch.Tensor],
    ) -> "PTCGEncoderBackbone":
        backbone = cls(config, card_feature_table, attack_feature_table)
        expected = backbone.state_dict()
        missing = sorted(set(expected) - set(state_dict))
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(f"missing encoder tensors: {preview}")
        incompatible = [
            name
            for name, tensor in expected.items()
            if tuple(state_dict[name].shape) != tuple(tensor.shape)
        ]
        if incompatible:
            preview = ", ".join(incompatible[:8])
            raise ValueError(f"shape-incompatible encoder tensors: {preview}")
        backbone.load_state_dict(
            {name: state_dict[name] for name in expected}, strict=True
        )
        return backbone


class PTCGValueModel(torch.nn.Module):
    """Predict acting-player value from the encoded global token."""

    def __init__(
        self,
        backbone: PTCGEncoderBackbone,
        *,
        head_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if type(head_layers) is not int or head_layers < 1:
            raise ValueError("head_layers must be an integer >= 1")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.backbone = backbone
        hidden: list[torch.nn.Module] = []
        for _ in range(head_layers - 1):
            hidden.extend(
                (
                    torch.nn.Linear(
                        backbone.config.d_model, backbone.config.d_model
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(float(dropout)),
                )
            )
        self.hidden = torch.nn.Sequential(*hidden)
        self.output = torch.nn.Linear(backbone.config.d_model, 1)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, *encoder_inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.backbone.encode_state(*encoder_inputs)
        global_token = encoded[GLOBAL_TOKEN_INDEX]
        return torch.tanh(self.output(self.hidden(global_token))).squeeze(-1)
