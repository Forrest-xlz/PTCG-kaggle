"""Train the expert state-action binary classifier from YAML configuration."""
from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_classifier.config import load_settings, load_train_yaml, project_path
from expert_classifier.data import build_classifier_indices, select_recent_dates
from expert_classifier.model import ExpertActionClassifier
from expert_classifier.runtime import (
    bootstrap_cg,
    build_backbone,
    build_lr_scheduler,
    feature_signature,
    forward_cached_batch,
    load_bc_weights,
    load_torch_checkpoint,
    model_config_from_checkpoint,
    model_config_from_train,
    resolve_device,
)
from training.expert_validation import load_expert_date_info
from training.feature_cache import IndexBatch, MmapFeatureDataset
from training.precision import PrecisionContext


def positive_weight(labels: torch.Tensor) -> float:
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        raise ValueError("both expert and non-expert training samples are required")
    return negatives / positives


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predicted = logits >= 0
    truth = labels >= 0.5
    tp = int((predicted & truth).sum().item())
    tn = int((~predicted & ~truth).sum().item())
    fp = int((predicted & ~truth).sum().item())
    fn = int((~predicted & truth).sum().item())
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def _evaluate(model, dataset, indices, labels_by_id, batch_size, device, precision):
    model.eval()
    loss_sum = 0.0
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    with torch.inference_mode():
        for index_batch in dataset.iter_index_batches(
            indices, batch_size, seed=0, shuffle=False
        ):
            batch = dataset.collate(index_batch)
            labels = torch.from_numpy(labels_by_id[index_batch.global_ids]).to(
                device=device, dtype=torch.float32
            )
            with precision.autocast():
                logits = forward_cached_batch(model, batch, device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss_sum += float(loss.item()) * labels.numel()
            logits_all.append(logits.float().cpu())
            labels_all.append(labels.cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return {"loss": loss_sum / labels.numel(), **binary_metrics(logits, labels)}


def main() -> None:
    settings = load_settings()
    train_raw = load_train_yaml(settings)
    bootstrap_cg(train_raw)
    from cg.api import all_attack, all_card_data

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    cards, attacks = all_card_data(), all_attack()

    if settings.model.init_mode == "checkpoint":
        source = load_torch_checkpoint(project_path(settings.model.checkpoint))
        config = model_config_from_checkpoint(source)
    else:
        source = None
        config = model_config_from_train(
            train_raw,
            max(card.cardId for card in cards) + 1,
            max(attack.attackId for attack in attacks) + 1,
        )
    backbone = build_backbone(config, cards, attacks)
    if source is not None:
        load_bc_weights(backbone, source)
    model = ExpertActionClassifier(backbone)

    train_section = train_raw["train"]
    dataset = MmapFeatureDataset(
        project_path(train_section["data"]),
        expected_signature=feature_signature(config),
    )
    output = project_path(settings.output)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        dates = select_recent_dates(dataset.shard_dates, settings.recent_dates)
        expert_info = load_expert_date_info(
            project_path(train_section["replay_episodes"]), dates, settings.expert_ratio
        )
        split = build_classifier_indices(
            dataset,
            dates,
            {date: info.expert_episode_ids for date, info in expert_info.items()},
            settings.validation_ratio,
            settings.seed,
        )
        labels_by_id = np.zeros(len(dataset), dtype=np.float32)
        labels_by_id[split.train_indices] = split.train_labels
        labels_by_id[split.validation_indices] = split.validation_labels
        pos_weight_value = positive_weight(torch.from_numpy(split.train_labels))
        for date in dates:
            info = expert_info[date]
            print(
                f"expert_date={date[0]}.{date[1]} cutoff={info.cutoff:g} "
                f"episodes={info.episode_count:,} expert_episodes={info.expert_episode_count:,}",
                flush=True,
            )
        print(
            f"classifier_train={split.train_indices.size:,} "
            f"validation={split.validation_indices.size:,} "
            f"positive_weight={pos_weight_value:.3f}", flush=True
        )

        device = resolve_device(settings.device)
        precision = PrecisionContext(settings.precision, device)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            betas=(settings.beta1, settings.beta2),
        )
        steps_per_epoch = math.ceil(split.train_indices.size / settings.batch_size)
        scheduler = build_lr_scheduler(
            optimizer, steps_per_epoch * settings.epochs, settings.warmup_steps
        )
        pos_weight_tensor = torch.tensor(pos_weight_value, device=device)
        best_loss = float("inf")
        global_step = 0
        for epoch in range(1, settings.epochs + 1):
            model.train()
            for index_batch in dataset.iter_index_batches(
                split.train_indices,
                settings.batch_size,
                seed=settings.seed + epoch,
                shuffle=True,
            ):
                batch = dataset.collate(index_batch)
                labels = torch.from_numpy(labels_by_id[index_batch.global_ids]).to(
                    device=device, dtype=torch.float32
                )
                optimizer.zero_grad(set_to_none=True)
                with precision.autocast():
                    logits = forward_cached_batch(model, batch, device)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, pos_weight=pos_weight_tensor
                    )
                step = precision.backward_step(
                    loss, model, optimizer, scheduler, settings.grad_clip_norm
                )
                if step.optimizer_stepped:
                    global_step += 1
                if global_step and global_step % settings.log_every_steps == 0:
                    metrics = binary_metrics(logits.detach(), labels)
                    print(
                        f"epoch={epoch} step={global_step:,} loss={loss.item():.4f} "
                        f"acc={metrics['accuracy']:.3f} precision={metrics['precision']:.3f} "
                        f"recall={metrics['recall']:.3f}", flush=True
                    )
            validation = _evaluate(
                model, dataset, split.validation_indices, labels_by_id,
                settings.batch_size, device, precision
            )
            print(f"validation epoch={epoch} " + " ".join(
                f"{name}={value:.4f}" for name, value in validation.items()
            ), flush=True)
            payload = {
                "model": model.state_dict(),
                "config": config.to_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": precision.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "validation": validation,
                "expert_classifier": asdict(settings),
            }
            torch.save(payload, checkpoint_dir / f"epoch-{epoch:03d}.pt")
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                torch.save(payload, output / "best.pt")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
