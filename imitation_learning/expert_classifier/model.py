"""Binary expert classifier built on decoded policy action features."""
from __future__ import annotations

import torch


class ExpertActionClassifier(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.expert_head = torch.nn.Linear(backbone.config.d_model, 1)

    def forward(
        self,
        *backbone_args,
        target: torch.Tensor,
        **backbone_kwargs,
    ) -> torch.Tensor:
        hidden = self.backbone.action_hidden(
            *backbone_args, **backbone_kwargs
        )
        if target.ndim != 1 or target.size(0) != hidden.size(0):
            raise ValueError("target must contain one action index per sample")
        if torch.any(target < 0) or torch.any(target >= hidden.size(1)):
            raise ValueError("target contains an invalid action index")
        selected = hidden[
            torch.arange(hidden.size(0), device=hidden.device), target
        ]
        return self.expert_head(selected).squeeze(-1)
