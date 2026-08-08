"""Extract reusable team and complete-Deck rows from replay ZIP archives."""
from __future__ import annotations

import csv
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
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "cfg" / "deck_trend.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deck.extract import extract_decks


SCHEMA_VERSION = 1
CSV_COLUMNS = [
    "date",
    "episode_id",
    "player",
    "team_name",
    "opponent_team_name",
    "deck",
    "reward",
    "result",
]


@dataclass(frozen=True)
class TrendExtractSettings:
    input: str
    output: str
    workers: int
    limit_members: int | None
    force: bool


def load_settings(path: Path = CONFIG_PATH) -> TrendExtractSettings:
    if not path.is_file():
        raise FileNotFoundError(f"Deck trend config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("extract"), dict):
        raise ValueError("deck_trend.yaml must contain an 'extract' mapping")
    try:
        settings = TrendExtractSettings(**raw["extract"])
    except TypeError as exc:
        raise ValueError(f"invalid extract configuration: {exc}") from exc
    if not isinstance(settings.input, str) or not settings.input.strip():
        raise ValueError("extract.input must be a non-empty string")
    if not isinstance(settings.output, str) or not settings.output.strip():
        raise ValueError("extract.output must be a non-empty string")
    if type(settings.workers) is not int or settings.workers < 1:
        raise ValueError("extract.workers must be an integer >= 1")
    if settings.limit_members is not None and (
        type(settings.limit_members) is not int or settings.limit_members < 1
    ):
        raise ValueError("extract.limit_members must be null or an integer >= 1")
    if not isinstance(settings.force, bool):
        raise ValueError("extract.force must be true or false")
    return settings


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _clean_team_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip()
    return name if name and name != "?" else None


def extract_team_names(payload: dict[str, Any]) -> list[str]:
    """Return two stable team names, preferring info.TeamNames."""
    info = payload.get("info") or {}
    team_names = info.get("TeamNames") or []
    agents = info.get("Agents") or []
    if not isinstance(team_names, (list, tuple)):
        team_names = []
    if not isinstance(agents, (list, tuple)):
        agents = []
    result: list[str] = []
    for player in range(2):
        preferred = (
            _clean_team_name(team_names[player])
            if player < len(team_names)
            else None
        )
        fallback = None
        if player < len(agents) and isinstance(agents[player], dict):
            fallback = _clean_team_name(agents[player].get("Name"))
        result.append(preferred or fallback or "")
    if any(not name for name in result):
        raise ValueError("replay does not contain two valid team names")
    return result


def _episode_id(payload: dict[str, Any], member_name: str) -> str:
    info = payload.get("info") or {}
    return str(info.get("EpisodeId") or Path(member_name).stem)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_gzip_csv(path: Path, rows: list[tuple[Any, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    try:
        with gzip.open(
            temporary,
            "wt",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _archive_fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def process_archive(
    args: tuple[str, str, bool, int | None],
) -> dict[str, Any]:
    archive_s, output_s, force, limit = args
    archive = Path(archive_s)
    output = Path(output_s)
    date_label = archive.stem
    shard = output / f"{date_label}.players.csv.gz"
    meta = output / f"{date_label}.meta.json"
    archive_size, archive_mtime_ns = _archive_fingerprint(archive)

    if shard.is_file() and meta.is_file() and not force:
        try:
            cached = json.loads(meta.read_text(encoding="utf-8"))
            reusable = (
                cached.get("schema_version") == SCHEMA_VERSION
                and cached.get("complete") is True
                and cached.get("archive_size") == archive_size
                and cached.get("archive_mtime_ns") == archive_mtime_ns
            )
            if reusable:
                return {
                    "archive": archive.name,
                    "status": "skipped",
                    "processed": int(cached.get("processed", 0)),
                    "player_rows": int(cached.get("player_rows", 0)),
                    "failed": int(cached.get("failed", 0)),
                    "complete": True,
                }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    rows: list[tuple[Any, ...]] = []
    errors: list[dict[str, str]] = []
    processed = 0
    with zipfile.ZipFile(archive) as replay_zip:
        members = [
            member
            for member in replay_zip.infolist()
            if not member.is_dir() and member.filename.endswith(".json")
        ]
        members.sort(key=lambda member: member.filename)
        if limit is not None:
            members = members[:limit]
        for member in members:
            try:
                with replay_zip.open(member) as raw:
                    payload = json.load(
                        io.TextIOWrapper(raw, encoding="utf-8")
                    )
                decks = extract_decks(payload)
                team_names = extract_team_names(payload)
                episode_id = _episode_id(payload, member.filename)
                rewards = payload.get("rewards") or [0, 0]
                for player in range(2):
                    reward = (
                        float(rewards[player] or 0)
                        if player < len(rewards)
                        else 0.0
                    )
                    result = (
                        "win"
                        if reward > 0
                        else "loss"
                        if reward < 0
                        else "draw"
                    )
                    rows.append(
                        (
                            date_label,
                            episode_id,
                            player,
                            team_names[player],
                            team_names[1 - player],
                            json.dumps(
                                sorted(decks[player]),
                                separators=(",", ":"),
                            ),
                            reward,
                            result,
                        )
                    )
                processed += 1
            except Exception as exc:
                errors.append(
                    {"member": member.filename, "error": str(exc)}
                )

    complete = limit is None
    _atomic_gzip_csv(shard, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "archive": archive.name,
        "archive_size": archive_size,
        "archive_mtime_ns": archive_mtime_ns,
        "limit_members": limit,
        "complete": complete,
        "processed": processed,
        "player_rows": len(rows),
        "failed": len(errors),
        "errors": errors[:100],
    }
    _atomic_text(
        meta,
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    return {**summary, "status": "written"}


def main() -> None:
    settings = load_settings()
    input_path = project_path(settings.input)
    output_path = project_path(settings.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Replay input does not exist: {input_path}")
    if input_path.is_file() and input_path.suffix.lower() != ".zip":
        raise ValueError(f"Replay input file must be a ZIP: {input_path}")
    archives = (
        [input_path]
        if input_path.is_file()
        else sorted(input_path.glob("*.zip"))
    )
    if not archives:
        raise FileNotFoundError(f"No ZIP archives found under: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    workers = min(settings.workers, len(archives))
    jobs = [
        (
            str(archive),
            str(output_path),
            settings.force,
            settings.limit_members,
        )
        for archive in archives
    ]
    print(
        f"config={CONFIG_PATH} archives={len(archives)} workers={workers} "
        f"limit_members={settings.limit_members} force={settings.force}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_archive, job) for job in jobs]
        for future in as_completed(futures):
            print(
                json.dumps(future.result(), ensure_ascii=False),
                flush=True,
            )


if __name__ == "__main__":
    main()
