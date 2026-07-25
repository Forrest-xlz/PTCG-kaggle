"""Mixed-precision policy and optimizer-step handling."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    grad_norm: float
    optimizer_stepped: bool
    grad_scale: float


class PrecisionContext:
    """Validate precision settings and keep AMP behavior in one place."""

    def __init__(self, name: str, device: torch.device):
        name = str(name).lower()
        if name not in {"fp32", "fp16", "bf16"}:
            raise ValueError("train.precision must be fp32, fp16, or bf16")
        if name != "fp32" and device.type != "cuda":
            raise ValueError(f"{name} mixed precision requires CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA precision selected but CUDA is unavailable")
        if name == "bf16":
            device_index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            with torch.cuda.device(device_index):
                bf16_supported = torch.cuda.is_bf16_supported()
            if not bf16_supported:
                raise ValueError(
                    "bf16 selected but the CUDA device does not support BF16"
                )

        self.name = name
        self.device = device
        self.autocast_enabled = name != "fp32"
        self.autocast_dtype = {
            "fp32": None,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[name]
        self.scaler = self._make_scaler(name == "fp16")

    @staticmethod
    def _make_scaler(enabled: bool):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def autocast(self):
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=True,
        )

    def backward_step(
        self,
        loss: torch.Tensor,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        grad_clip_norm: float,
    ) -> OptimizerStepResult:
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_norm
            )
            old_scale = float(self.scaler.get_scale())
            self.scaler.step(optimizer)
            self.scaler.update()
            new_scale = float(self.scaler.get_scale())
            optimizer_stepped = new_scale >= old_scale
            if optimizer_stepped:
                scheduler.step()
            return OptimizerStepResult(
                grad_norm=float(grad_norm.item()),
                optimizer_stepped=optimizer_stepped,
                grad_scale=new_scale,
            )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip_norm
        )
        optimizer.step()
        scheduler.step()
        return OptimizerStepResult(
            grad_norm=float(grad_norm.item()),
            optimizer_stepped=True,
            grad_scale=1.0,
        )
