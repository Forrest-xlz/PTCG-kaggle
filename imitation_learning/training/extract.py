"""Convert replay ZIPs to resumable gzip JSONL training shards."""
from __future__ import annotations

import gzip
import hashlib
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
SCHEMA_VERSION = 5


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


def _iter_player_records(
    steps: list,
    player: int,
    episode,
    date: str,
    deck: list[int],
    player_result: str,
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
            "player_result": player_result,
            "observation": observation,
            "selected": action,
        }


def _player_results(
    rewards: list | tuple,
    player_count: int,
) -> tuple[str, ...]:
    winning_players = {
        player
        for player, reward in enumerate(rewards)
        if reward is not None and float(reward) > 0
    }
    if not winning_players:
        return ("draw",) * player_count
    return tuple(
        "win" if player in winning_players else "loss"
        for player in range(player_count)
    )


def _exact_deck_id(deck: list[int]) -> str:
    canonical = ",".join(map(str, sorted(map(int, deck))))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()
    return f"D-{digest[:12]}"


def _annotate_auxiliary_labels(
    records_by_player: dict[int, list[dict]],
    decks: list[list[int]],
    final_own_prize_counts: list[int],
) -> None:
    deck_ids = [_exact_deck_id(deck) for deck in decks]
    for player, records in records_by_player.items():
        if not records:
            continue
        opponent = 1 - player
        for index, record in enumerate(records):
            has_next = index + 1 < len(records)
            next_select = (
                records[index + 1]["observation"].get("select") or {}
                if has_next
                else {}
            )
            record["next_decision_valid"] = has_next
            record["next_select_type"] = int(next_select.get("type") or 0)
            record["next_select_context"] = int(next_select.get("context") or 0)
            record["opponent_deck_id"] = deck_ids[opponent]
            record["final_own_prize_count"] = final_own_prize_counts[player]


def _final_own_prize_counts(steps: list, player_count: int) -> list[int]:
    counts: list[int] = []
    for player in range(player_count):
        for step in reversed(steps):
            if player >= len(step):
                continue
            observation = step[player].get("observation") or {}
            current = observation.get("current") or {}
            players = current.get("players") or []
            try:
                yours = int(current["yourIndex"])
                counts.append(len(players[yours].get("prize") or []))
                break
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        else:
            raise ValueError(f"player {player} has no valid prize observation")
    return counts


def process_archive(job):
    archive = Path(job[0])
    output = Path(job[1])
    force, limit = job[2], job[3]
    shard = output / f"{archive.stem}.jsonl.gz"
    meta = output / f"{archive.stem}.meta.json"
    if shard.exists() and meta.exists() and not force:
        try:
            cached = json.loads(meta.read_text(encoding="utf-8"))
            if (
                cached.get("schema_version") == SCHEMA_VERSION
                and cached.get("player_results") == "win-loss-draw"
                and cached.get("auxiliary_labels")
                == "next-decision-opponent-deck-final-prize-v1"
            ):
                return {"archive": archive.name, "status": "skipped"}
        except (OSError, json.JSONDecodeError):
            pass

    output.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(dir=output, suffix=".tmp")
    os.close(file_descriptor)
    episodes = samples = failed = draws = 0
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
                    player_results = _player_results(rewards, len(decks))
                    if player_results and all(
                        result == "draw" for result in player_results
                    ):
                        draws += 1
                    episode = replay.get("info", {}).get(
                        "EpisodeId", Path(member.filename).stem
                    )
                    steps = replay.get("steps", [])
                    final_own_prizes = _final_own_prize_counts(
                        steps, len(decks)
                    )
                    records_by_player = {}
                    for player in range(len(decks)):
                        if player >= len(decks):
                            continue
                        records_by_player[player] = list(_iter_player_records(
                            steps,
                            player,
                            episode,
                            archive.stem,
                            decks[player],
                            player_results[player],
                        ))
                    _annotate_auxiliary_labels(
                        records_by_player, decks, final_own_prizes
                    )
                    for player in range(len(decks)):
                        for record in records_by_player.get(player, []):
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
        "player_results": "win-loss-draw",
        "auxiliary_labels": "next-decision-opponent-deck-final-prize-v1",
        "archive": archive.name,
        "status": "written",
        "episodes": episodes,
        "samples": samples,
        "failed": failed,
        "draws": draws,
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
