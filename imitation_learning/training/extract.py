"""Convert replay ZIPs to resumable gzip JSONL training shards."""
from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml

from deck.extract import extract_decks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "cfg" / "extract.yaml"


@dataclass(frozen=True)
class ExtractSettings:
    input: str
    output: str
    workers: int
    limit_members: int | None
    force: bool


def load_settings(path: Path = CONFIG_PATH) -> ExtractSettings:
    """Load and validate the fixed hierarchical extraction configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Extraction config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("extract"), dict):
        raise ValueError("extract.yaml must contain an 'extract' mapping")
    settings = ExtractSettings(**raw["extract"])
    if settings.workers < 1:
        raise ValueError("extract.workers must be >= 1")
    if settings.limit_members is not None and settings.limit_members < 1:
        raise ValueError("extract.limit_members must be null or >= 1")
    return settings


def project_path(value: str) -> Path:
    """Resolve relative YAML paths from the imitation_learning project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _selected(action):
    if isinstance(action, list) and all(isinstance(index, int) for index in action):
        return action
    return None


def process_archive(job):
    archive = Path(job[0])
    output = Path(job[1])
    force, limit = job[2], job[3]
    shard = output / f"{archive.stem}.jsonl.gz"
    meta = output / f"{archive.stem}.meta.json"
    if shard.exists() and meta.exists() and not force:
        return {"archive": archive.name, "status": "skipped"}

    output.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(dir=output, suffix=".tmp")
    os.close(file_descriptor)
    episodes = samples = failed = 0
    errors = []
    try:
        with (
            gzip.open(temporary_path, "wt", encoding="utf-8") as destination,
            zipfile.ZipFile(archive) as source,
        ):
            members = [
                member
                for member in source.infolist()
                if not member.is_dir() and member.filename.endswith(".json")
            ]
            if limit is not None:
                members = members[:limit]

            for member in members:
                try:
                    with source.open(member) as raw:
                        replay = json.load(io.TextIOWrapper(raw, encoding="utf-8"))
                    decks = extract_decks(replay)
                    episode = replay.get("info", {}).get(
                        "EpisodeId", Path(member.filename).stem
                    )
                    for step_index, step in enumerate(replay.get("steps", [])):
                        for player, state in enumerate(step):
                            observation = state.get("observation")
                            action = _selected(state.get("action"))
                            if (
                                not observation
                                or observation.get("current") is None
                                or action is None
                            ):
                                continue
                            record = {
                                "episode_id": episode,
                                "date": archive.stem,
                                "step": step_index,
                                "player": player,
                                "deck": decks[player],
                                "observation": observation,
                                "selected": action,
                            }
                            destination.write(
                                json.dumps(record, separators=(",", ":")) + "\n"
                            )
                            samples += 1
                    episodes += 1
                except Exception as exc:
                    failed += 1
                    if len(errors) < 100:
                        errors.append(
                            {"member": member.filename, "error": str(exc)}
                        )
        os.replace(temporary_path, shard)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    summary = {
        "archive": archive.name,
        "status": "written",
        "episodes": episodes,
        "samples": samples,
        "failed": failed,
        "errors": errors,
    }
    meta.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    settings = load_settings()
    input_path = project_path(settings.input)
    output_path = project_path(settings.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Replay input does not exist: {input_path}")

    archives = (
        [input_path]
        if input_path.is_file()
        else sorted(input_path.glob("*.zip"))
    )
    if not archives:
        raise FileNotFoundError(f"No ZIP archives found under: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    print(
        f"config={CONFIG_PATH} archives={len(archives)} workers={settings.workers} "
        f"limit_members={settings.limit_members} force={settings.force}"
    )

    jobs = [
        (
            str(archive),
            str(output_path),
            settings.force,
            settings.limit_members,
        )
        for archive in archives
    ]
    with ProcessPoolExecutor(max_workers=settings.workers) as pool:
        futures = [pool.submit(process_archive, job) for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

