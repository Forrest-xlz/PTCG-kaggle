"""Train the notebook transformer with replay actions as supervision."""
from __future__ import annotations

import gzip
import json
import math
import multiprocessing as mp
import queue
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "cfg" / "train.yaml"

# Configure local project imports before importing cg/model modules. `cg_path`
# must point to the parent directory containing the `cg` package.
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Training config not found: {CONFIG_PATH}")
with CONFIG_PATH.open("r", encoding="utf-8") as _config_handle:
    _bootstrap_config = yaml.safe_load(_config_handle)
_cg_value = (_bootstrap_config or {}).get("train", {}).get("cg_path")
if not _cg_value:
    raise ValueError("train.cg_path is required in cfg/train.yaml")
_cg_path = Path(_cg_value)
if not _cg_path.is_absolute():
    _cg_path = PROJECT_ROOT / _cg_path
_cg_path = _cg_path.resolve()
if _cg_path.name == "cg":
    _cg_path = _cg_path.parent
if not (_cg_path / "cg" / "__init__.py").exists():
    raise FileNotFoundError(
        f"train.cg_path must contain the cg package; not found under: {_cg_path}"
    )

# Support both module execution and direct script execution.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(_cg_path) not in sys.path:
    sys.path.insert(0, str(_cg_path))

from cg.api import SelectContext, all_attack, all_card_data, to_observation_class
from model.features import decoder_features, encoder_features, enumerate_actions
from model.network import ModelConfig, PTCGTransformer


MAX_ACTIONS = 64


@dataclass(frozen=True)
class TrainSettings:
    cg_path: str
    data: str
    output: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_ratio: float
    max_samples: int | None
    seed: int
    device: str
    preload: bool
    preload_workers: int
    preload_chunk_size: int
    shuffle_buffer: int
    log_every_steps: int
    grad_clip_norm: float


@dataclass(frozen=True)
class ModelSettings:
    d_model: int
    ffn_multiplier: int
    num_heads: int
    encoder_layers: int
    decoder_layers: int


@dataclass(frozen=True)
class WandbSettings:
    enabled: bool
    project: str
    group: str
    name: str
    mode: str = "online"


@dataclass(frozen=True)
class ExperimentSettings:
    version_name: str
    train: TrainSettings
    model: ModelSettings
    wandb: WandbSettings


def load_settings(path: Path = CONFIG_PATH) -> ExperimentSettings:
    """Load and validate the hierarchical YAML training configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping")
    missing = {"version_name", "train", "model", "wandb"} - raw.keys()
    if missing:
        raise ValueError(f"Missing YAML sections: {sorted(missing)}")

    version_name = str(raw["version_name"]).strip()
    if not version_name:
        raise ValueError("version_name must not be empty")

    def interpolate(value):
        if isinstance(value, str):
            return value.replace("${version_name}", version_name)
        if isinstance(value, dict):
            return {key: interpolate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [interpolate(item) for item in value]
        return value

    raw = interpolate(raw)

    settings = ExperimentSettings(
        version_name=version_name,
        train=TrainSettings(**raw["train"]),
        model=ModelSettings(**raw["model"]),
        wandb=WandbSettings(**raw["wandb"]),
    )
    train, model = settings.train, settings.model
    if train.epochs < 1 or train.batch_size < 1:
        raise ValueError("train.epochs and train.batch_size must be >= 1")
    if train.preload_workers < 1 or train.preload_chunk_size < 1:
        raise ValueError("preload worker and chunk settings must be >= 1")
    if train.shuffle_buffer < 2 or train.log_every_steps < 1:
        raise ValueError("shuffle_buffer must be >= 2 and log_every_steps >= 1")
    if train.learning_rate <= 0 or train.weight_decay < 0 or train.grad_clip_norm <= 0:
        raise ValueError("learning rate/grad clip must be positive and weight decay non-negative")
    if not 0 <= train.warmup_ratio < 1:
        raise ValueError("train.warmup_ratio must be in [0, 1)")
    if not 0 <= train.beta1 < 1 or not 0 <= train.beta2 < 1:
        raise ValueError("train.beta1 and train.beta2 must be in [0, 1)")
    if model.d_model < 1 or model.ffn_multiplier <= 0:
        raise ValueError("model.d_model and model.ffn_multiplier must be positive")
    if model.num_heads < 1 or model.d_model % model.num_heads != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if model.encoder_layers < 1 or model.decoder_layers < 1:
        raise ValueError("encoder_layers and decoder_layers must be >= 1")
    if settings.wandb.enabled and not settings.wandb.project:
        raise ValueError("wandb.project is required when wandb.enabled is true")
    return settings


def project_path(value: str) -> Path:
    """Resolve relative YAML paths from the imitation_learning project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def metadata_sample_count(root: Path) -> int:
    """Read the extractor sidecars to plan scheduler steps without loading data."""
    total = 0
    for path in root.glob("*.meta.json"):
        with path.open("r", encoding="utf-8") as handle:
            total += int(json.load(handle).get("samples", 0))
    if total <= 0:
        raise ValueError(
            "Could not infer samples per epoch from metadata; set train.max_samples "
            "or enable preload"
        )
    return total


def build_lr_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    """Linear warmup to target LR, followed by cosine decay to zero."""
    if total_steps < 1:
        raise ValueError("total training steps must be >= 1")
    warmup_steps = min(total_steps - 1, round(total_steps * warmup_ratio))

    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier), warmup_steps


def sequential_records(root: Path):
    """Read all shards sequentially without retaining decoded JSON records."""
    for path in sorted(root.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def records(root: Path, shuffle_buffer: int = 10_000):
    """Bounded-memory shuffle over all date shards."""
    buffer = []
    for record in sequential_records(root):
        buffer.append(record)
        if len(buffer) >= shuffle_buffer:
            random.shuffle(buffer)
            while len(buffer) > shuffle_buffer // 2:
                yield buffer.pop()
    random.shuffle(buffer)
    yield from buffer


def merge_sparse(vectors, device, words_per_sample: int | None = None):
    """Concatenate SparseVectors and optionally pad each to a fixed word count."""
    indices: list[int] = []
    values: list[float] = []
    offsets: list[int] = []

    for vector in vectors:
        base = len(indices)
        indices.extend(vector.index)
        values.extend(vector.value)
        offsets.extend(base + offset for offset in vector.offset)
        if words_per_sample is not None:
            if len(vector.offset) > words_per_sample:
                raise ValueError(
                    f"Sparse vector has {len(vector.offset)} words; maximum is {words_per_sample}"
                )
            # Repeated offsets create empty EmbeddingBag entries for padding.
            offsets.extend([len(indices)] * (words_per_sample - len(vector.offset)))

    return (
        torch.tensor(indices, dtype=torch.int64, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
        torch.tensor(offsets, dtype=torch.int64, device=device),
    )


def prepare_sample(record, config):
    """Convert one replay record to model features, or return None if unrepresentable."""
    obs = to_observation_class(record["observation"])
    actions = enumerate_actions(
        len(obs.select.option), obs.select.maxCount, limit=MAX_ACTIONS
    )
    try:
        target = actions.index(record["selected"])
    except ValueError:
        # The recorded action is outside the notebook's first-64 candidate cap.
        return None

    return {
        "encoder": encoder_features(obs, record["deck"], config.card_count),
        "decoder": decoder_features(
            obs, actions, config.card_count, config.attack_count
        ),
        "target": target,
        "action_count": len(actions),
    }


def prepared_samples(root, config, shuffle_buffer):
    """Stream shuffled records and construct features just before training."""
    for record in records(root, shuffle_buffer=shuffle_buffer):
        sample = prepare_sample(record, config)
        if sample is not None:
            yield sample


def preload_samples(root, config, max_samples: int | None = None):
    """Precompute the dataset once and retain only sparse model features in RAM."""
    samples = []
    skipped = 0
    for record in sequential_records(root):
        sample = prepare_sample(record, config)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)
        if len(samples) % 10_000 == 0:
            print(
                f"preload samples={len(samples):,} skipped={skipped:,}",
                flush=True,
            )
        if max_samples is not None and len(samples) >= max_samples:
            break
    print(
        f"preload complete samples={len(samples):,} skipped={skipped:,}",
        flush=True,
    )
    return samples


def _parallel_preload_worker(
    worker_id,
    paths,
    config_dict,
    output_queue,
    stop_event,
    chunk_size,
):
    """Read assigned shards and send prepared-feature chunks to the parent."""
    config = ModelConfig(**config_dict)
    chunk = []
    skipped = 0
    try:
        for path_string in paths:
            if stop_event.is_set():
                break
            path = Path(path_string)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if stop_event.is_set():
                        break
                    sample = prepare_sample(json.loads(line), config)
                    if sample is None:
                        skipped += 1
                        continue
                    chunk.append(sample)
                    if len(chunk) >= chunk_size:
                        output_queue.put(("chunk", worker_id, chunk))
                        chunk = []
        if chunk and not stop_event.is_set():
            output_queue.put(("chunk", worker_id, chunk))
    except Exception:
        output_queue.put(("error", worker_id, traceback.format_exc()))
    finally:
        output_queue.put(("done", worker_id, skipped))


def parallel_preload_samples(
    root,
    config,
    workers,
    chunk_size,
    max_samples: int | None = None,
):
    """Precompute shards in parallel and collect sparse features in RAM."""
    paths = sorted(root.glob("*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"No .jsonl.gz shards found under {root}")
    workers = min(workers, len(paths))
    assignments = [[] for _ in range(workers)]
    for index, path in enumerate(paths):
        assignments[index % workers].append(str(path))

    # Spawn is required on Windows and keeps behavior consistent across OSes.
    context = mp.get_context("spawn")
    output_queue = context.Queue(maxsize=max(2, workers * 2))
    stop_event = context.Event()
    processes = [
        context.Process(
            target=_parallel_preload_worker,
            args=(
                worker_id,
                assigned_paths,
                config.to_dict(),
                output_queue,
                stop_event,
                chunk_size,
            ),
        )
        for worker_id, assigned_paths in enumerate(assignments)
    ]

    samples = []
    skipped = 0
    completed_workers = set()
    error = None
    for process in processes:
        process.start()
    try:
        while len(completed_workers) < len(processes):
            try:
                message_type, worker_id, payload = output_queue.get(timeout=5)
            except queue.Empty:
                crashed = [
                    index
                    for index, process in enumerate(processes)
                    if index not in completed_workers
                    and not process.is_alive()
                    and process.exitcode is not None
                ]
                if crashed:
                    error = f"preload workers exited without completion: {crashed}"
                    stop_event.set()
                    break
                continue
            if message_type == "chunk":
                chunk = payload
                if max_samples is not None:
                    remaining = max_samples - len(samples)
                    if remaining <= 0:
                        stop_event.set()
                        continue
                    chunk = chunk[:remaining]
                samples.extend(chunk)
                if len(samples) % 10_000 < len(chunk):
                    print(
                        f"parallel preload samples={len(samples):,} "
                        f"workers_done={len(completed_workers)}/{len(processes)}",
                        flush=True,
                    )
                if max_samples is not None and len(samples) >= max_samples:
                    stop_event.set()
            elif message_type == "error":
                error = f"preload worker {worker_id} failed:\n{payload}"
                stop_event.set()
            elif message_type == "done":
                skipped += int(payload)
                completed_workers.add(worker_id)
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        output_queue.close()

    if error is not None:
        raise RuntimeError(error)
    print(
        f"parallel preload complete samples={len(samples):,} "
        f"skipped={skipped:,} workers={workers}",
        flush=True,
    )
    return samples


def train_batch(batch, model, optimizer, scheduler, device, grad_clip_norm):
    """Run one optimizer step over a padded variable-action batch."""
    encoder_inputs = merge_sparse(
        [sample["encoder"] for sample in batch], device
    )
    decoder_inputs = merge_sparse(
        [sample["decoder"] for sample in batch],
        device,
        words_per_sample=MAX_ACTIONS,
    )

    policy_logits = model(*encoder_inputs, *decoder_inputs)
    action_counts = torch.tensor(
        [sample["action_count"] for sample in batch], device=device
    )
    invalid_actions = (
        torch.arange(MAX_ACTIONS, device=device).unsqueeze(0)
        >= action_counts.unsqueeze(1)
    )
    policy_logits = policy_logits.masked_fill(
        invalid_actions, torch.finfo(policy_logits.dtype).min
    )

    policy_targets = torch.tensor(
        [sample["target"] for sample in batch], dtype=torch.long, device=device
    )
    loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets)

    learning_rate = optimizer.param_groups[0]["lr"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    scheduler.step()

    correct = int((policy_logits.argmax(dim=1) == policy_targets).sum().item())
    return float(loss.item()), correct, float(grad_norm.item()), learning_rate


def main():
    settings = load_settings()
    train_cfg, model_cfg, wandb_cfg = (
        settings.train,
        settings.model,
        settings.wandb,
    )
    data_path = project_path(train_cfg.data)
    output_path = project_path(train_cfg.output)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_path}")

    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    output_path.mkdir(parents=True, exist_ok=True)

    cards = all_card_data()
    config = ModelConfig(
        card_count=max(card.cardId for card in cards) + 1,
        attack_count=max(attack.attackId for attack in all_attack()) + 1,
        recover_special_condition=int(SelectContext.RECOVER_SPECIAL_CONDITION),
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        d_feedforward=int(model_cfg.d_model * model_cfg.ffn_multiplier),
        encoder_layers=model_cfg.encoder_layers,
        decoder_layers=model_cfg.decoder_layers,
    )
    if train_cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg.device)
    model = PTCGTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )
    history = []
    print(
        f"version={settings.version_name} config={CONFIG_PATH} "
        f"device={device} batch_size={train_cfg.batch_size} "
        f"encoder_layers={model_cfg.encoder_layers} "
        f"decoder_layers={model_cfg.decoder_layers} d_model={model_cfg.d_model} "
        f"d_feedforward={config.d_feedforward} heads={model_cfg.num_heads}"
    )

    wandb_run = None
    if wandb_cfg.enabled:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb.enabled is true but wandb is not installed; run pip install -r requirements.txt"
            ) from exc
        wandb_run = wandb.init(
            project=wandb_cfg.project,
            group=wandb_cfg.group or None,
            name=wandb_cfg.name or None,
            mode=wandb_cfg.mode,
            config={
                "version_name": settings.version_name,
                "train": asdict(train_cfg),
                "model": asdict(model_cfg),
                "network": config.to_dict(),
            },
        )

    in_memory_samples = None
    preload_start = time.perf_counter()
    if train_cfg.preload:
        print(
            "Preloading and precomputing samples. Memory usage grows with the "
            "number of samples; set train.max_samples for a bounded trial.",
            flush=True,
        )
        if train_cfg.preload_workers == 1:
            in_memory_samples = preload_samples(
                data_path, config, max_samples=train_cfg.max_samples
            )
        else:
            in_memory_samples = parallel_preload_samples(
                data_path,
                config,
                workers=train_cfg.preload_workers,
                chunk_size=train_cfg.preload_chunk_size,
                max_samples=train_cfg.max_samples,
            )
        if not in_memory_samples:
            raise RuntimeError("No representable training samples were loaded")
    preload_seconds = time.perf_counter() - preload_start
    if wandb_run is not None:
        wandb_run.log(
            {
                "data/preload_seconds": preload_seconds,
                "data/preloaded_samples": len(in_memory_samples or []),
            }
        )

    if in_memory_samples is not None:
        samples_per_epoch = len(in_memory_samples)
    else:
        available_samples = metadata_sample_count(data_path)
        samples_per_epoch = (
            available_samples
            if train_cfg.max_samples is None
            else min(available_samples, train_cfg.max_samples)
        )
    steps_per_epoch = math.ceil(samples_per_epoch / train_cfg.batch_size)
    total_steps = steps_per_epoch * train_cfg.epochs
    scheduler, warmup_steps = build_lr_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=train_cfg.warmup_ratio
    )
    print(
        f"samples_per_epoch={samples_per_epoch:,} steps_per_epoch={steps_per_epoch:,} "
        f"total_steps={total_steps:,} warmup_steps={warmup_steps:,}"
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "schedule/steps_per_epoch": steps_per_epoch,
                "schedule/total_steps": total_steps,
                "schedule/warmup_steps": warmup_steps,
                "schedule/warmup_ratio": train_cfg.warmup_ratio,
            }
        )

    global_step = 0
    total_training_start = time.perf_counter()
    for epoch in range(train_cfg.epochs):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        batch = []

        if in_memory_samples is not None:
            random.shuffle(in_memory_samples)
            epoch_samples = iter(in_memory_samples)
        else:
            epoch_samples = prepared_samples(
                data_path, config, shuffle_buffer=train_cfg.shuffle_buffer
            )

        for sample in epoch_samples:
            if (
                train_cfg.max_samples is not None
                and count + len(batch) >= train_cfg.max_samples
            ):
                break
            batch.append(sample)
            if len(batch) < train_cfg.batch_size:
                continue

            loss, batch_correct, grad_norm, learning_rate = train_batch(
                batch,
                model,
                optimizer,
                scheduler,
                device,
                grad_clip_norm=train_cfg.grad_clip_norm,
            )
            batch_count = len(batch)
            total_loss += loss * batch_count
            correct += batch_correct
            count += batch_count
            global_step += 1
            batch = []
            if global_step % train_cfg.log_every_steps == 0:
                elapsed = max(time.perf_counter() - epoch_start, 1e-9)
                metrics = {
                    "train/batch_loss": loss,
                    "train/batch_accuracy": batch_correct / batch_count,
                    "train/running_loss": total_loss / count,
                    "train/running_accuracy": correct / count,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": learning_rate,
                    "train/samples": count,
                    "train/samples_per_second": count / elapsed,
                    "train/epoch": epoch + 1,
                    "train/optimizer_step": global_step,
                }
                print(
                    f"epoch={epoch + 1} samples={count} "
                    f"loss={metrics['train/running_loss']:.4f} "
                    f"acc={metrics['train/running_accuracy']:.3f} "
                    f"samples/s={metrics['train/samples_per_second']:.1f}"
                )
                if wandb_run is not None:
                    wandb_run.log(metrics)

        # Do not discard the final partial batch.
        if batch:
            loss, batch_correct, grad_norm, learning_rate = train_batch(
                batch,
                model,
                optimizer,
                scheduler,
                device,
                grad_clip_norm=train_cfg.grad_clip_norm,
            )
            batch_count = len(batch)
            total_loss += loss * batch_count
            correct += batch_correct
            count += batch_count
            global_step += 1

        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch + 1,
            "samples": count,
            "batch_size": train_cfg.batch_size,
            "loss": total_loss / max(count, 1),
            "accuracy": correct / max(count, 1),
            "seconds": epoch_seconds,
            "samples_per_second": count / max(epoch_seconds, 1e-9),
        }
        history.append(row)
        print(row)
        epoch_metrics = {
            "epoch/loss": row["loss"],
            "epoch/accuracy": row["accuracy"],
            "epoch/samples": row["samples"],
            "epoch/seconds": row["seconds"],
            "epoch/samples_per_second": row["samples_per_second"],
            "epoch/index": epoch + 1,
            "epoch/optimizer_step": global_step,
            "epoch/learning_rate": optimizer.param_groups[0]["lr"],
        }
        if device.type == "cuda":
            epoch_metrics["system/gpu_peak_memory_gb"] = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
            torch.cuda.reset_peak_memory_stats(device)
        if wandb_run is not None:
            wandb_run.log(epoch_metrics)
        torch.save(
            {
                "model": model.state_dict(),
                "config": config.to_dict(),
                "experiment": {
                    "version_name": settings.version_name,
                    "train": asdict(train_cfg),
                    "model": asdict(model_cfg),
                    "wandb": asdict(wandb_cfg),
                },
                "history": history,
            },
            output_path / f"epoch-{epoch + 1}.pt",
        )

    (output_path / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.summary["training/total_seconds"] = (
            time.perf_counter() - total_training_start
        )
        wandb_run.summary["training/final_loss"] = history[-1]["loss"]
        wandb_run.summary["training/final_accuracy"] = history[-1]["accuracy"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
