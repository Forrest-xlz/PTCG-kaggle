"""Convert replay ZIPs to resumable gzip JSONL training shards."""
from __future__ import annotations

import gzip
import io
import json
import os
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml

# Support both `python -m training.extract` and direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deck.extract import extract_decks


CONFIG_PATH = PROJECT_ROOT / "cfg" / "extract.yaml"
SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ExtractSettings:
    input: str
    output: str
    workers: int
    winner_only: bool
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
    if not isinstance(settings.winner_only, bool):
        raise ValueError("extract.winner_only must be true or false")
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


def _iter_player_records(
    steps: list,
    player: int,
    episode,
    date: str,
    deck: list[int],
):
    """Pair each active observation with the action recorded in the next step."""
    for step_index in range(max(0, len(steps) - 1)):
        step = steps[step_index]
        next_step = steps[step_index + 1]
        if player >= len(step) or player >= len(next_step):
            continue
        state = step[player]
        if state.get("status") != "ACTIVE":
            continue
        observation = state.get("observation")
        action = _selected(next_step[player].get("action"))
        if (
            not observation
            or observation.get("current") is None
            or action is None
        ):
            continue
        yield {
            "episode_id": episode,
            "date": date,
            "step": step_index,
            "player": player,
            "deck": deck,
            "observation": observation,
            "selected": action,
        }


def process_archive(job):
    archive = Path(job[0])
    output = Path(job[1])
    force, limit, winner_only = job[2], job[3], job[4]
    shard = output / f"{archive.stem}.jsonl.gz"
    meta = output / f"{archive.stem}.meta.json"
    if shard.exists() and meta.exists() and not force:
        try:
            cached = json.loads(meta.read_text(encoding="utf-8"))
            if (
                cached.get("schema_version") == SCHEMA_VERSION
                and cached.get("winner_only") == winner_only
            ):
                return {"archive": archive.name, "status": "skipped"}
        except (OSError, json.JSONDecodeError):
            pass

    output.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(dir=output, suffix=".tmp")
    os.close(file_descriptor)
    episodes = samples = failed = no_winner = 0
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
                    rewards = replay.get("rewards") or []
                    winning_players = [
                        player
                        for player, reward in enumerate(rewards)
                        if reward is not None and float(reward) > 0
                    ]
                    if winner_only and not winning_players:
                        no_winner += 1
                        episodes += 1
                        continue
                    selected_players = (
                        winning_players
                        if winner_only
                        else list(range(len(decks)))
                    )
                    episode = replay.get("info", {}).get(
                        "EpisodeId", Path(member.filename).stem
                    )
                    steps = replay.get("steps", [])
                    for player in selected_players:
                        if player >= len(decks):
                            continue
                        for record in _iter_player_records(
                            steps,
                            player,
                            episode,
                            archive.stem,
                            decks[player],
                        ):
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
        "schema_version": SCHEMA_VERSION,
        "winner_only": winner_only,
        "archive": archive.name,
        "status": "written",
        "episodes": episodes,
        "samples": samples,
        "failed": failed,
        "no_winner": no_winner,
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
        f"winner_only={settings.winner_only} limit_members={settings.limit_members} "
        f"force={settings.force}"
    )

    jobs = [
        (
            str(archive),
            str(output_path),
            settings.force,
            settings.limit_members,
            settings.winner_only,
        )
        for archive in archives
    ]
    with ProcessPoolExecutor(max_workers=settings.workers) as pool:
        futures = [pool.submit(process_archive, job) for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
