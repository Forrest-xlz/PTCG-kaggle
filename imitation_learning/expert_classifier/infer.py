"""Score losing samples and save stable identities above a probability threshold."""
from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_classifier.config import load_settings, load_train_yaml, project_path
from expert_classifier.data import select_recent_dates
from expert_classifier.model import ExpertActionClassifier
from expert_classifier.runtime import (
    bootstrap_cg,
    build_backbone,
    feature_signature,
    forward_cached_batch,
    load_torch_checkpoint,
    model_config_from_checkpoint,
    resolve_device,
)
from training.feature_cache import (
    PLAYER_RESULT_LOSS,
    MmapFeatureDataset,
)
from training.precision import PrecisionContext


def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(probabilities) >= float(threshold)


def select_loser_indices(
    dataset: MmapFeatureDataset, dates: tuple[tuple[int, int], ...]
) -> np.ndarray:
    selected = set(dates)
    dtype = np.uint32 if len(dataset) <= np.iinfo(np.uint32).max else np.uint64
    parts: list[np.ndarray] = []
    for shard_id, (shard, date) in enumerate(zip(dataset.shards, dataset.shard_dates)):
        if date not in selected:
            continue
        ids = np.arange(dataset.starts[shard_id], dataset.ends[shard_id], dtype=dtype)
        parts.append(ids[shard.arrays["player_result"] == PLAYER_RESULT_LOSS])
    return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)


def _identity_rows(dataset: MmapFeatureDataset, global_ids: np.ndarray):
    shard_ids = np.searchsorted(dataset.ends, global_ids, side="right")
    for global_id, shard_id in zip(global_ids, shard_ids):
        local = int(global_id) - int(dataset.starts[int(shard_id)])
        shard = dataset.shards[int(shard_id)]
        date = dataset.shard_dates[int(shard_id)]
        yield (
            date,
            int(shard.arrays["episode_id"][local]),
            int(shard.arrays["player"][local]),
            int(shard.arrays["step"][local]),
        )


def main() -> None:
    settings = load_settings()
    train_raw = load_train_yaml(settings)
    bootstrap_cg(train_raw)
    from cg.api import all_attack, all_card_data

    checkpoint_path = project_path(settings.inference.checkpoint)
    checkpoint = load_torch_checkpoint(checkpoint_path)
    config = model_config_from_checkpoint(checkpoint)
    backbone = build_backbone(config, all_card_data(), all_attack())
    model = ExpertActionClassifier(backbone)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("expert classifier checkpoint model must be a mapping")
    model.load_state_dict(state)

    dataset = MmapFeatureDataset(
        project_path(train_raw["train"]["data"]),
        expected_signature=feature_signature(config),
    )
    output = project_path(settings.inference.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "selected_loser_samples.csv.gz"
    summary_path = output / "inference_summary.json"
    try:
        dates = select_recent_dates(dataset.shard_dates, settings.inference.recent_dates)
        indices = select_loser_indices(dataset, dates)
        if indices.size == 0:
            raise ValueError("no losing samples found in inference dates")
        device = resolve_device(settings.device)
        precision = PrecisionContext(settings.precision, device)
        model.to(device).eval()
        counts = defaultdict(lambda: {"examined": 0, "retained": 0})
        retained_total = 0
        with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "episode_id", "player", "step", "expert_probability"])
            with torch.inference_mode():
                for index_batch in dataset.iter_index_batches(
                    indices, settings.batch_size, seed=0, shuffle=False
                ):
                    batch = dataset.collate(index_batch)
                    with precision.autocast():
                        logits = forward_cached_batch(model, batch, device)
                    probabilities = torch.sigmoid(logits).float().cpu().numpy()
                    keep = apply_threshold(probabilities, settings.inference.threshold)
                    for identity, probability, selected in zip(
                        _identity_rows(dataset, index_batch.global_ids), probabilities, keep
                    ):
                        date, episode_id, player, step = identity
                        key = f"{date[0]}.{date[1]}"
                        counts[key]["examined"] += 1
                        if selected:
                            writer.writerow([key, episode_id, player, step, f"{float(probability):.8f}"])
                            counts[key]["retained"] += 1
                            retained_total += 1
        summary = {
            "checkpoint": str(checkpoint_path),
            "threshold": settings.inference.threshold,
            "recent_dates": settings.inference.recent_dates,
            "dates": [f"{month}.{day}" for month, day in dates],
            "examined": int(indices.size),
            "retained": retained_total,
            "retained_ratio": retained_total / int(indices.size),
            "by_date": dict(counts),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for date, values in counts.items():
            print(
                f"inference_date={date} losses={values['examined']:,} "
                f"retained={values['retained']:,} "
                f"ratio={values['retained'] / max(values['examined'], 1):.3%}",
                flush=True,
            )
        print(f"selection={csv_path} retained={retained_total:,}/{indices.size:,}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
