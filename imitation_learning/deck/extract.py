"""Extract the two initial decks from replay ZIPs in parallel."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="ZIP file or directory of ZIPs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(5, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--limit-members", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    ns = parser.parse_args()
    archives = [ns.input] if ns.input.is_file() else sorted(ns.input.glob("*.zip"))
    ns.output.mkdir(parents=True, exist_ok=True)
    jobs = [(str(x), str(ns.output), ns.force, ns.limit_members) for x in archives]
    with ProcessPoolExecutor(max_workers=ns.workers) as pool:
        futures = [pool.submit(process_archive, job) for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False))


if __name__ == "__main__":
    main()
