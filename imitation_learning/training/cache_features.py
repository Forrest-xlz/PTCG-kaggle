"""Convert winner-only replay JSONL shards into packed mmap feature caches."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "cfg" / "cache.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Cache config not found: {CONFIG_PATH}")
with CONFIG_PATH.open("r", encoding="utf-8") as _handle:
    _bootstrap = yaml.safe_load(_handle) or {}
_cg_value = _bootstrap.get("cache", {}).get("cg_path")
if not _cg_value:
    raise ValueError("cache.cg_path is required in cfg/cache.yaml")
_cg_path = Path(_cg_value)
if not _cg_path.is_absolute():
    _cg_path = (PROJECT_ROOT / _cg_path).resolve()
if _cg_path.name == "cg":
    _cg_path = _cg_path.parent
if not (_cg_path / "cg" / "__init__.py").exists():
    raise FileNotFoundError(
        f"cache.cg_path must contain the cg package; not found under: {_cg_path}"
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(_cg_path) not in sys.path:
    sys.path.insert(0, str(_cg_path))

from cg.api import SelectContext, all_attack, all_card_data, to_observation_class
from model.features import decoder_features, encoder_features, enumerate_actions
from model.network import ModelConfig
from training.feature_cache import (
    CACHE_SCHEMA_VERSION,
    MAX_ACTIONS,
    FeatureRecord,
    PackedShard,
    PackedShardWriter,
    stable_episode_key,
)


@dataclass(frozen=True)
class CacheSettings:
    cg_path: str
    input: str
    output: str
    workers: int
    samples_per_shard: int
    force: bool


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(path: Path = CONFIG_PATH) -> CacheSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cache"), dict):
        raise ValueError("cfg/cache.yaml must contain a cache mapping")
    settings = CacheSettings(**raw["cache"])
    if settings.workers < 1:
        raise ValueError("cache.workers must be >= 1")
    if settings.samples_per_shard < 1:
        raise ValueError("cache.samples_per_shard must be >= 1")
    if not isinstance(settings.force, bool):
        raise ValueError("cache.force must be true or false")
    return settings


def feature_signature(config: ModelConfig) -> dict:
    return {
        "card_count": config.card_count,
        "attack_count": config.attack_count,
        "encoder_size": config.encoder_size,
        "num_encoder_words": config.num_encoder_words,
        "decoder_size": config.decoder_size,
        "recover_special_condition": config.recover_special_condition,
        "max_actions": MAX_ACTIONS,
        "action_enumeration": "max-to-min-v1",
    }


def _prepare_record(
    record: dict, config: ModelConfig
) -> tuple[FeatureRecord | None, str | None]:
    obs = to_observation_class(record["observation"])
    actions = enumerate_actions(
        len(obs.select.option),
        obs.select.minCount,
        obs.select.maxCount,
        limit=MAX_ACTIONS,
    )
    selected = sorted(record["selected"])
    if not obs.select.minCount <= len(selected) <= obs.select.maxCount:
        return None, "invalid_selected_count"
    if (
        len(set(selected)) != len(selected)
        or any(index < 0 or index >= len(obs.select.option) for index in selected)
    ):
        return None, "invalid_selected_index"
    try:
        target = actions.index(selected)
    except ValueError:
        return None, "outside_first_64"
    encoder = encoder_features(obs, record["deck"], config.card_count)
    decoder = decoder_features(
        obs, actions, config.card_count, config.attack_count
    )
    if any(value != 1.0 for value in decoder.value):
        raise ValueError("decoder feature values are no longer all one")
    return (
        FeatureRecord(
            encoder_index=encoder.index,
            encoder_value=encoder.value,
            encoder_offset=encoder.offset,
            decoder_index=decoder.index,
            decoder_offset=decoder.offset,
            target=target,
            action_count=len(actions),
            episode_key=stable_episode_key(record["episode_id"]),
        ),
        None,
    )


def prepare_record(record: dict, config: ModelConfig) -> FeatureRecord | None:
    """Prepare one record; return None when its expert action is unsupported."""
    return _prepare_record(record, config)[0]


def _source_stem(path: Path) -> str:
    suffix = ".jsonl.gz"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def _source_meta(path: Path) -> dict:
    meta = path.with_name(f"{_source_stem(path)}.meta.json")
    if not meta.exists():
        raise FileNotFoundError(f"extract metadata not found for {path.name}: {meta}")
    payload = json.loads(meta.read_text(encoding="utf-8"))
    if payload.get("winner_only") is not True:
        raise ValueError(
            f"{path.name} was not extracted with winner_only=true; rerun extraction"
        )
    if int(payload.get("schema_version", 0)) < 3:
        raise ValueError(
            f"{path.name} uses the old action alignment; rerun training.extract"
        )
    return payload


def _manifest_path(output: Path, stem: str) -> Path:
    return output / f"{stem}.source.json"


def _part_paths(output: Path, stem: str) -> list[Path]:
    return sorted(
        path for path in output.glob(f"{stem}.part-*.cache") if path.is_dir()
    )


def _clear_source(output: Path, stem: str) -> None:
    for path in _part_paths(output, stem):
        shutil.rmtree(path)
    manifest = _manifest_path(output, stem)
    if manifest.exists():
        manifest.unlink()
    for path in output.glob(f".{stem}.part-*.cache.*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _source_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_is_current(
    source: Path,
    output: Path,
    signature: dict,
    samples_per_shard: int,
) -> bool:
    manifest_path = _manifest_path(output, _source_stem(source))
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            return False
        if manifest.get("source") != _source_identity(source):
            return False
        if manifest.get("signature") != signature:
            return False
        if manifest.get("samples_per_shard") != samples_per_shard:
            return False
        parts = manifest.get("parts", [])
        if not isinstance(parts, list):
            return False
        counted = 0
        for name in parts:
            shard = PackedShard(output / name, expected_signature=signature)
            counted += len(shard)
            shard.close()
        return counted == int(manifest.get("samples", -1))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def process_source(job) -> dict:
    source = Path(job[0])
    output = Path(job[1])
    config = ModelConfig(**job[2])
    signature = job[3]
    samples_per_shard = int(job[4])
    force = bool(job[5])
    stem = _source_stem(source)
    _source_meta(source)

    if not force and _cache_is_current(
        source, output, signature, samples_per_shard
    ):
        return {"source": source.name, "status": "skipped"}

    _clear_source(output, stem)
    writer = None
    part_index = 0
    part_samples = 0
    samples = 0
    skipped = 0
    skip_reasons = {
        "invalid_selected_count": 0,
        "invalid_selected_index": 0,
        "outside_first_64": 0,
    }
    part_names = []
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    prepared, skip_reason = _prepare_record(
                        json.loads(line), config
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"{source.name}:{line_number}: {exc}"
                    ) from exc
                if prepared is None:
                    skipped += 1
                    skip_reasons[skip_reason] += 1
                    continue
                if writer is None:
                    part_name = f"{stem}.part-{part_index:05d}.cache"
                    writer = PackedShardWriter(
                        output / part_name,
                        signature=signature,
                        source={
                            **_source_identity(source),
                            "part": part_index,
                        },
                    )
                    part_samples = 0
                writer.add(prepared)
                part_samples += 1
                samples += 1
                if part_samples >= samples_per_shard:
                    writer.finalize()
                    part_names.append(part_name)
                    writer = None
                    part_index += 1
        if writer is not None:
            writer.finalize()
            part_names.append(part_name)
            writer = None
    except Exception:
        if writer is not None:
            writer.abort()
        _clear_source(output, stem)
        raise

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": _source_identity(source),
        "signature": signature,
        "samples_per_shard": samples_per_shard,
        "samples": samples,
        "skipped": skipped,
        "skip_reasons": skip_reasons,
        "parts": part_names,
    }
    _atomic_json(_manifest_path(output, stem), manifest)
    return {
        "source": source.name,
        "status": "written",
        "samples": samples,
        "skipped": skipped,
        "skip_reasons": skip_reasons,
        "parts": len(part_names),
    }


def main() -> None:
    settings = load_settings()
    input_path = project_path(settings.input)
    output_path = project_path(settings.output)
    sources = sorted(input_path.glob("*.jsonl.gz"))
    if not sources:
        raise FileNotFoundError(f"No .jsonl.gz files found under {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    cards = all_card_data()
    config = ModelConfig(
        card_count=max(card.cardId for card in cards) + 1,
        attack_count=max(attack.attackId for attack in all_attack()) + 1,
        recover_special_condition=int(SelectContext.RECOVER_SPECIAL_CONDITION),
    )
    signature = feature_signature(config)
    jobs = [
        (
            str(source),
            str(output_path),
            config.to_dict(),
            signature,
            settings.samples_per_shard,
            settings.force,
        )
        for source in sources
    ]
    workers = min(settings.workers, len(jobs))
    results = []
    print(
        f"config={CONFIG_PATH} sources={len(sources)} workers={workers} "
        f"samples_per_shard={settings.samples_per_shard:,}",
        flush=True,
    )
    if workers == 1:
        for job in jobs:
            result = process_source(job)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_source, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)

    manifest_samples = 0
    manifest_parts = 0
    part_names = []
    manifest_skip_reasons = {
        "invalid_selected_count": 0,
        "invalid_selected_index": 0,
        "outside_first_64": 0,
    }
    for source in sources:
        manifest = json.loads(
            _manifest_path(output_path, _source_stem(source)).read_text(
                encoding="utf-8"
            )
        )
        manifest_samples += int(manifest.get("samples", 0))
        source_parts = manifest.get("parts", [])
        manifest_parts += len(source_parts)
        part_names.extend(source_parts)
        for reason, count in manifest.get("skip_reasons", {}).items():
            if reason in manifest_skip_reasons:
                manifest_skip_reasons[reason] += int(count)
    summary = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "settings": asdict(settings),
        "signature": signature,
        "sources": len(sources),
        "written": sum(result["status"] == "written" for result in results),
        "skipped": sum(result["status"] == "skipped" for result in results),
        "parts": manifest_parts,
        "part_names": part_names,
        "samples": manifest_samples,
        "skip_reasons": manifest_skip_reasons,
    }
    _atomic_json(output_path / "cache.meta.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
