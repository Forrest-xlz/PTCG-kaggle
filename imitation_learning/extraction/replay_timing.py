"""Extract per-player timing metrics from one dated replay archive."""
from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "cfg" / "extract_replay_timing.yaml"
DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})$")
MANIFEST_COLUMNS = {"episode_id", "avg_score", "min_score", "sum_score", "agent_count"}
PLAYER_COLUMNS = (
    "date",
    "episode_id",
    "player_index",
    "team_name",
    "startup_time_seconds",
    "subsequent_time_seconds",
    "subsequent_action_count",
    "mean_step_time_seconds",
    "avg_score",
    "min_score",
    "max_score",
    "sum_score",
)


@dataclass(frozen=True)
class ExtractionSettings:
    input: Path
    output: Path
    date: str
    force: bool


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def parse_month_day(value: str) -> tuple[int, int]:
    match = DATE_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"Invalid replay date {value!r}; expected M.D")
    month, day = map(int, match.groups())
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        raise ValueError(f"Invalid replay date {value!r}; expected M.D")
    return month, day


def load_settings(path: Path = CONFIG_PATH, project_root: Path = PROJECT_ROOT) -> ExtractionSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("extract"), dict):
        raise ValueError("Configuration must contain an extract mapping")
    cfg = raw["extract"]
    allowed = {"input", "output", "date", "force"}
    unknown = set(cfg) - allowed
    missing = allowed - set(cfg)
    if unknown:
        raise ValueError(f"Unsupported extract settings: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing extract settings: {sorted(missing)}")
    date = str(cfg["date"]).strip()
    if date != "latest":
        parse_month_day(date)
    if type(cfg["force"]) is not bool:
        raise ValueError("extract.force must be boolean")
    return ExtractionSettings(
        input=_resolve_path(str(cfg["input"]), project_root),
        output=_resolve_path(str(cfg["output"]), project_root),
        date=date,
        force=cfg["force"],
    )


def resolve_archive(input_dir: Path, date: str) -> tuple[str, Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Replay input directory not found: {input_dir}")
    archives: dict[str, Path] = {}
    for path in input_dir.glob("*.zip"):
        try:
            parse_month_day(path.stem)
        except ValueError:
            continue
        archives[path.stem] = path
    if not archives:
        raise FileNotFoundError(f"No replay archives named M.D.zip found in {input_dir}")
    resolved = max(archives, key=parse_month_day) if date == "latest" else date
    if resolved not in archives:
        raise FileNotFoundError(f"Replay archive for {resolved} not found in {input_dir}")
    return resolved, archives[resolved]


def _find_manifest_member(archive: zipfile.ZipFile) -> str:
    names = [name for name in archive.namelist() if PurePosixPath(name).name == "manifest.csv"]
    if len(names) != 1:
        raise ValueError(f"{archive.filename} must contain exactly one manifest.csv; found {len(names)}")
    return names[0]


def _load_manifest(archive_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(_find_manifest_member(archive)) as raw:
            manifest = pd.read_csv(
                io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""),
                dtype={"episode_id": "string"},
            )
    missing = MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest.csv is missing columns: {sorted(missing)}")
    manifest = manifest.copy()
    manifest["episode_id"] = manifest["episode_id"].str.strip()
    if manifest["episode_id"].isna().any() or manifest["episode_id"].eq("").any():
        raise ValueError("manifest.csv contains empty episode_id values")
    if manifest["episode_id"].duplicated().any():
        raise ValueError("manifest.csv contains duplicate episode_id values")
    for column in ("avg_score", "min_score", "sum_score", "agent_count"):
        manifest[column] = pd.to_numeric(manifest[column], errors="raise")
    if not np.isfinite(manifest[["avg_score", "min_score", "sum_score"]]).all().all():
        raise ValueError("manifest.csv contains non-finite scores")
    if not manifest["agent_count"].eq(2).all():
        raise ValueError("Replay timing extraction requires exactly two agents per replay")
    manifest["max_score"] = manifest["sum_score"] - manifest["min_score"]
    if manifest["max_score"].lt(manifest["min_score"]).any():
        raise ValueError("manifest.csv has inconsistent min_score and sum_score")
    return manifest


def _finite_remaining_time(agent_state: object) -> float | None:
    if not isinstance(agent_state, dict) or not isinstance(agent_state.get("observation"), dict):
        return None
    value = agent_state["observation"].get("remainingOverageTime")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def extract_player_timings(payload: dict[str, Any], date: str) -> list[dict[str, Any]]:
    info = payload.get("info")
    steps = payload.get("steps")
    if not isinstance(info, dict) or not isinstance(info.get("TeamNames"), list) or not isinstance(steps, list):
        return []
    episode_id = str(info.get("EpisodeId") or "").strip()
    rows: list[dict[str, Any]] = []
    for player_index, raw_team_name in enumerate(info["TeamNames"][:2]):
        team_name = str(raw_team_name or "").strip()
        if not team_name:
            continue
        remaining = [
            value
            for step in steps
            if isinstance(step, list) and player_index < len(step)
            for value in [_finite_remaining_time(step[player_index])]
            if value is not None
        ]
        positive_deltas = [a - b for a, b in zip(remaining, remaining[1:]) if a - b > 1e-9]
        if not positive_deltas:
            continue
        later = positive_deltas[1:]
        later_total = float(sum(later))
        rows.append(
            {
                "date": date,
                "episode_id": episode_id,
                "player_index": player_index,
                "team_name": team_name,
                "startup_time_seconds": float(positive_deltas[0]),
                "subsequent_time_seconds": later_total,
                "subsequent_action_count": len(later),
                "mean_step_time_seconds": later_total / len(later) if later else np.nan,
            }
        )
    return rows


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".csv", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _valid_cache(path: Path) -> bool:
    try:
        cached = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    return set(PLAYER_COLUMNS).issubset(cached.columns)


def extract_archive(archive_path: Path, output_dir: Path, date: str, force: bool) -> dict[str, Any]:
    cache_path = output_dir / f"{date}.player_timings.csv"
    error_path = output_dir / f"{date}.extraction_errors.csv"
    if cache_path.exists() and not force and _valid_cache(cache_path):
        row_count = len(pd.read_csv(cache_path, usecols=["episode_id"]))
        return {"status": "skipped", "player_rows": row_count, "failed_members": 0}

    manifest = _load_manifest(archive_path)
    manifest_lookup = manifest.set_index("episode_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir() and PurePosixPath(member.filename).suffix.lower() == ".json"]
        for member in members:
            try:
                with archive.open(member) as raw:
                    payload = json.load(io.TextIOWrapper(raw, encoding="utf-8"))
                info = payload.get("info") if isinstance(payload, dict) else None
                episode_id = str((info or {}).get("EpisodeId") or PurePosixPath(member.filename).stem)
                manifest_values = manifest_lookup.get(episode_id)
                if manifest_values is None:
                    raise KeyError(f"episode_id={episode_id} is absent from manifest.csv")
                extracted = extract_player_timings(payload, date)
                if not extracted:
                    raise ValueError("no valid player timing rows")
                for row in extracted:
                    row["episode_id"] = episode_id
                    row.update({key: float(manifest_values[key]) for key in ("avg_score", "min_score", "max_score", "sum_score")})
                    rows.append(row)
            except Exception as error:
                errors.append({"member": member.filename, "error": f"{type(error).__name__}: {error}"})
    if not rows:
        raise ValueError("No valid replay-player timing rows were extracted")
    result = pd.DataFrame(rows).loc[:, PLAYER_COLUMNS]
    _atomic_csv(result, cache_path)
    if errors:
        _atomic_csv(pd.DataFrame(errors), error_path)
    else:
        error_path.unlink(missing_ok=True)
    return {"status": "written", "player_rows": len(result), "failed_members": len(errors)}


def main(config_path: Path = CONFIG_PATH) -> None:
    settings = load_settings(config_path)
    date, archive = resolve_archive(settings.input, settings.date)
    summary = extract_archive(archive, settings.output, date, settings.force)
    print(f"date={date} archive={archive.name} status={summary['status']} player_rows={summary['player_rows']:,} failed_members={summary['failed_members']:,}")


if __name__ == "__main__":
    main()
