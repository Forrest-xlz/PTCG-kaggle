"""Extract the two initial decks from replay ZIPs in parallel."""
from __future__ import annotations

import csv
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
CONFIG_PATH = PROJECT_ROOT / "cfg" / "extract_deck_lists.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DeckExtractSettings:
    input: str
    output: str
    workers: int
    limit_members: int | None
    force: bool


def load_settings(path: Path = CONFIG_PATH) -> DeckExtractSettings:
    if not path.exists():
        raise FileNotFoundError(f"Deck extraction config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("extract"), dict):
        raise ValueError("deck_extract.yaml must contain an 'extract' mapping")
    settings = DeckExtractSettings(**raw["extract"])
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


def _episode_id(payload: dict[str, Any], member: str) -> str:
    return str(payload.get("info", {}).get("EpisodeId") or Path(member).stem)


def extract_decks(payload: dict[str, Any]) -> list[list[int]]:
    """Return the two complete initial decks from a replay payload."""
    steps = payload.get("steps") or []
    if not steps:
        raise ValueError("replay has no steps")
    for agent_state in steps[0]:
        visualizations = agent_state.get("visualize") or []
        for item in visualizations:
            action = item.get("action")
            if (
                isinstance(action, list)
                and len(action) == 2
                and all(isinstance(deck, list) for deck in action)
                and all(len(deck) == 60 for deck in action)
            ):
                return [[int(card) for card in deck] for deck in action]
    raise ValueError("initial 60-card decks not found")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def process_archive(args: tuple[str, str, bool, int | None]) -> dict[str, Any]:
    archive_s, output_s, force, limit = args
    archive, output = Path(archive_s), Path(output_s)
    stem = archive.stem
    shard, meta = output / f"{stem}.decks.csv", output / f"{stem}.meta.json"
    if shard.exists() and meta.exists() and not force:
        try:
            cached = json.loads(meta.read_text(encoding="utf-8"))
            if cached.get("schema_version") == SCHEMA_VERSION:
                return {"archive": archive.name, "status": "skipped"}
        except (OSError, json.JSONDecodeError):
            pass

    # One replay produces exactly two compact rows (one complete deck per player).
    rows: list[tuple[str, str, int, str, float, str]] = []
    errors: list[dict[str, str]] = []
    processed = 0
    with zipfile.ZipFile(archive) as zf:
        members = [x for x in zf.infolist() if not x.is_dir() and x.filename.endswith(".json")]
        if limit is not None:
            members = members[:limit]
        for member in members:
            try:
                with zf.open(member) as raw:
                    payload = json.load(io.TextIOWrapper(raw, encoding="utf-8"))
                episode = _episode_id(payload, member.filename)
                rewards = payload.get("rewards") or [0, 0]
                for player, deck in enumerate(extract_decks(payload)):
                    reward = float(rewards[player] or 0) if player < len(rewards) else 0.0
                    result = "win" if reward > 0 else "loss" if reward < 0 else "draw"
                    rows.append((stem, episode, player, json.dumps(sorted(deck), separators=(",", ":")), reward, result))
                processed += 1
            except Exception as exc:  # keep long-running extraction resumable
                errors.append({"member": member.filename, "error": str(exc)})

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["date", "episode_id", "player", "deck", "reward", "result"])
    writer.writerows(rows)
    _atomic_text(shard, buf.getvalue())
    summary = {"schema_version": SCHEMA_VERSION, "archive": archive.name, "processed": processed,
               "deck_rows": len(rows), "failed": len(errors), "errors": errors[:100]}
    _atomic_text(meta, json.dumps(summary, ensure_ascii=False, indent=2))
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
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
