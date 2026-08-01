"""Export a compact inference-only checkpoint for Kaggle submission."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


PRECISION_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

# Configure the export here; no command-line arguments are used.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "training" / "step-00007000.pt"
OUTPUT_PATH: Path | None = PROJECT_ROOT / "model_submission.pt"
PRECISION = "fp16"  # fp16, bf16, or fp32


@dataclass(frozen=True)
class ExportSummary:
    source: Path
    destination: Path
    precision: str
    source_bytes: int
    destination_bytes: int

    @property
    def reduction_ratio(self) -> float:
        if self.source_bytes == 0:
            return 0.0
        return 1.0 - self.destination_bytes / self.source_bytes


def _load_checkpoint(path: Path) -> Mapping:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch 2.0 does not expose the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("checkpoint must contain a model state mapping")
    if not isinstance(checkpoint.get("config"), Mapping):
        raise ValueError("checkpoint must contain a config mapping")
    return checkpoint


def export_inference_checkpoint(
    source: str | Path,
    destination: str | Path,
    precision: str = "fp16",
) -> ExportSummary:
    """Strip training state and cast floating model tensors for inference."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if source == destination:
        raise ValueError("destination must not overwrite the training checkpoint")
    if precision not in PRECISION_DTYPES:
        choices = ", ".join(PRECISION_DTYPES)
        raise ValueError(f"precision must be one of: {choices}")

    checkpoint = _load_checkpoint(source)
    dtype = PRECISION_DTYPES[precision]
    model_state = {
        name: (
            tensor.detach().to(device="cpu", dtype=dtype)
            if tensor.is_floating_point()
            else tensor.detach().cpu()
        )
        for name, tensor in checkpoint["model"].items()
    }
    payload = {
        "model": model_state,
        "config": dict(checkpoint["config"]),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return ExportSummary(
        source=source,
        destination=destination,
        precision=precision,
        source_bytes=source.stat().st_size,
        destination_bytes=destination.stat().st_size,
    )


def _default_destination(source: Path, precision: str) -> Path:
    suffix = source.suffix or ".pt"
    return source.with_name(f"{source.stem}.inference-{precision}{suffix}")


def _mib(size: int) -> float:
    return size / 1024**2


def main() -> None:
    destination = OUTPUT_PATH or _default_destination(
        CHECKPOINT_PATH, PRECISION
    )
    summary = export_inference_checkpoint(
        CHECKPOINT_PATH,
        destination,
        PRECISION,
    )
    print(f"Source:      {summary.source}")
    print(f"Output:      {summary.destination}")
    print(f"Precision:   {summary.precision}")
    print(f"Source size: {_mib(summary.source_bytes):.2f} MiB")
    print(f"Output size: {_mib(summary.destination_bytes):.2f} MiB")
    print(f"Reduction:   {summary.reduction_ratio:.1%}")


if __name__ == "__main__":
    main()
