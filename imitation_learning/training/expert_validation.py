"""Build per-date expert episode sets from replay archive manifests."""
from __future__ import annotations

import csv
import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from training.feature_cache import parse_source_date, stable_episode_key


REQUIRED_COLUMNS = {"episode_id", "min_score", "sum_score", "agent_count"}


@dataclass(frozen=True, slots=True)
class ExpertDateInfo:
    date: tuple[int, int]
    cutoff: float
    episode_count: int
    expert_episode_keys: frozenset[int]

    @property
    def expert_episode_count(self) -> int:
        return len(self.expert_episode_keys)


def _archive_by_date(root: Path) -> dict[tuple[int, int], Path]:
    archives: dict[tuple[int, int], Path] = {}
    for path in root.glob("*.zip"):
        try:
            date = parse_source_date(path.name)
        except ValueError:
            continue
        if date in archives:
            raise ValueError(
                f"multiple replay archives found for {date[0]}.{date[1]}: "
                f"{archives[date]} and {path}"
            )
        archives[date] = path
    return archives


def _read_manifest(
    archive_path: Path,
    date: tuple[int, int],
    ratio: float,
) -> ExpertDateInfo:
    with zipfile.ZipFile(archive_path) as archive:
        manifest_names = [
            name
            for name in archive.namelist()
            if PurePosixPath(name).name == "manifest.csv"
        ]
        if len(manifest_names) != 1:
            raise ValueError(
                f"{archive_path} must contain exactly one manifest.csv; "
                f"found {len(manifest_names)}"
            )
        with archive.open(manifest_names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{archive_path} manifest.csv is missing columns: "
                    f"{sorted(missing)}"
                )
            episodes: list[tuple[int, float]] = []
            scores: list[float] = []
            seen_ids: set[str] = set()
            for row_number, row in enumerate(reader, start=2):
                episode_id = str(row["episode_id"]).strip()
                if not episode_id:
                    raise ValueError(
                        f"{archive_path} manifest row {row_number} has no episode_id"
                    )
                if episode_id in seen_ids:
                    raise ValueError(
                        f"{archive_path} has duplicate episode_id {episode_id!r}"
                    )
                seen_ids.add(episode_id)
                try:
                    agent_count = int(row["agent_count"])
                    low_score = float(row["min_score"])
                    sum_score = float(row["sum_score"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{archive_path} manifest row {row_number} has invalid scores"
                    ) from exc
                high_score = sum_score - low_score
                if agent_count != 2:
                    raise ValueError(
                        f"{archive_path} manifest row {row_number} has "
                        f"agent_count={agent_count}, expected 2"
                    )
                if not math.isfinite(low_score) or not math.isfinite(high_score):
                    raise ValueError(
                        f"{archive_path} manifest row {row_number} has non-finite scores"
                    )
                if high_score < low_score:
                    raise ValueError(
                        f"{archive_path} manifest row {row_number} has inconsistent "
                        "min_score and sum_score"
                    )
                episodes.append((stable_episode_key(episode_id), high_score))
                scores.extend((low_score, high_score))

    if not episodes:
        raise ValueError(f"{archive_path} manifest.csv contains no episodes")
    top_count = max(1, math.ceil(len(scores) * ratio))
    cutoff = sorted(scores, reverse=True)[top_count - 1]
    expert_keys = frozenset(
        episode_key
        for episode_key, high_score in episodes
        if high_score >= cutoff
    )
    if not expert_keys:
        raise ValueError(f"{archive_path} produced an empty expert episode set")
    return ExpertDateInfo(
        date=date,
        cutoff=cutoff,
        episode_count=len(episodes),
        expert_episode_keys=expert_keys,
    )


def load_expert_date_info(
    replay_root: Path,
    required_dates: Iterable[tuple[int, int]],
    ratio: float,
) -> dict[tuple[int, int], ExpertDateInfo]:
    if not 0 < ratio <= 1:
        raise ValueError("expert validation ratio must be in (0, 1]")
    replay_root = Path(replay_root)
    if not replay_root.is_dir():
        raise FileNotFoundError(
            f"Replay archive directory not found: {replay_root}"
        )
    archives = _archive_by_date(replay_root)
    dates = sorted(set(required_dates))
    missing = [date for date in dates if date not in archives]
    if missing:
        labels = ", ".join(f"{month}.{day}" for month, day in missing)
        raise FileNotFoundError(
            f"Replay archives required by the cache are missing: {labels}"
        )
    return {
        date: _read_manifest(archives[date], date, ratio)
        for date in dates
    }
