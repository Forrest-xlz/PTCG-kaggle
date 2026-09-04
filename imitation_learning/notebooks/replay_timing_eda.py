"""Generate Replay Timing EDA figures and tables from extracted timing data."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "cfg" / "replay_timing_eda.yaml"
DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})$")
SCORE_COLUMNS = {"min": "min_score", "max": "max_score", "avg": "avg_score"}
FEATURE_COLUMNS = ("startup_time_mean_seconds", "mean_step_time_seconds")
REQUIRED_PLAYER_COLUMNS = {
    "episode_id",
    "team_name",
    "startup_time_seconds",
    "subsequent_time_seconds",
    "subsequent_action_count",
    "avg_score",
    "min_score",
    "max_score",
}


@dataclass(frozen=True)
class ScoreFilter:
    mode: str
    threshold: float


@dataclass(frozen=True)
class ClusteringSettings:
    n_clusters: int
    random_state: int
    n_init: int


@dataclass(frozen=True)
class AnalysisSettings:
    input: Path
    output: Path
    date: str
    score_filter: ScoreFilter
    clustering: ClusteringSettings


def _path(value: str, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _parse_date(value: str) -> tuple[int, int]:
    match = DATE_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"Invalid replay date {value!r}; expected M.D")
    month, day = map(int, match.groups())
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        raise ValueError(f"Invalid replay date {value!r}; expected M.D")
    return month, day


def load_settings(path: Path = CONFIG_PATH, project_root: Path = PROJECT_ROOT) -> AnalysisSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("analysis"), dict):
        raise ValueError("Configuration must contain an analysis mapping")
    cfg = raw["analysis"]
    allowed = {"input", "output", "date", "score_filter", "clustering"}
    if set(cfg) != allowed:
        raise ValueError(f"analysis settings must be exactly {sorted(allowed)}")
    score = cfg["score_filter"]
    clustering = cfg["clustering"]
    if not isinstance(score, dict) or set(score) != {"mode", "threshold"}:
        raise ValueError("score_filter must contain mode and threshold")
    if score["mode"] not in SCORE_COLUMNS:
        raise ValueError(f"Unsupported score mode: {score['mode']}")
    if not isinstance(clustering, dict) or set(clustering) != {"n_clusters", "random_state", "n_init"}:
        raise ValueError("clustering must contain n_clusters, random_state, and n_init")
    settings = ClusteringSettings(
        n_clusters=int(clustering["n_clusters"]),
        random_state=int(clustering["random_state"]),
        n_init=int(clustering["n_init"]),
    )
    if settings.n_clusters < 1 or settings.n_init < 1:
        raise ValueError("clustering n_clusters and n_init must be positive")
    date = str(cfg["date"]).strip()
    if date != "latest":
        _parse_date(date)
    return AnalysisSettings(
        input=_path(str(cfg["input"]), project_root),
        output=_path(str(cfg["output"]), project_root),
        date=date,
        score_filter=ScoreFilter(str(score["mode"]), float(score["threshold"])),
        clustering=settings,
    )


def resolve_timing_csv(input_dir: Path, date: str) -> tuple[str, Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Replay timing input directory not found: {input_dir}")
    suffix = ".player_timings.csv"
    candidates: dict[str, Path] = {}
    for path in input_dir.glob(f"*{suffix}"):
        label = path.name.removesuffix(suffix)
        try:
            _parse_date(label)
        except ValueError:
            continue
        candidates[label] = path
    if not candidates:
        raise FileNotFoundError(f"No player timing CSV files found in {input_dir}")
    resolved = max(candidates, key=_parse_date) if date == "latest" else date
    if resolved not in candidates:
        raise FileNotFoundError(f"Player timing CSV for {resolved} not found in {input_dir}")
    return resolved, candidates[resolved]


def aggregate_team_timings(players: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PLAYER_COLUMNS - set(players.columns)
    if missing:
        raise ValueError(f"Player timing CSV is missing columns: {sorted(missing)}")
    if players.empty:
        raise ValueError("Player timing CSV is empty")
    frame = players.copy()
    numeric = REQUIRED_PLAYER_COLUMNS - {"episode_id", "team_name"}
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    grouped = frame.groupby("team_name", as_index=False).agg(
        replay_count=("episode_id", "nunique"),
        startup_time_mean_seconds=("startup_time_seconds", "mean"),
        subsequent_time_seconds=("subsequent_time_seconds", "sum"),
        subsequent_action_count=("subsequent_action_count", "sum"),
        avg_score=("avg_score", "mean"),
        min_score=("min_score", "mean"),
        max_score=("max_score", "mean"),
    )
    grouped = grouped.loc[grouped["subsequent_action_count"].gt(0)].copy()
    if grouped.empty:
        raise ValueError("No teams contain subsequent actions")
    grouped["mean_step_time_seconds"] = grouped["subsequent_time_seconds"] / grouped["subsequent_action_count"]
    return grouped


def filter_team_timings(teams: pd.DataFrame, score_filter: ScoreFilter) -> pd.DataFrame:
    try:
        score_column = SCORE_COLUMNS[score_filter.mode]
    except KeyError as error:
        raise ValueError(f"Unsupported score mode: {score_filter.mode}") from error
    result = teams.copy()
    result["score_value"] = pd.to_numeric(result[score_column], errors="raise")
    return result.loc[result["score_value"].ge(score_filter.threshold)].copy()


def _log_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Timing features must be finite and nonnegative")
    return np.log1p(values)


def assign_timing_clusters(
    teams: pd.DataFrame,
    filtered: pd.DataFrame,
    settings: ClusteringSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if settings.n_clusters < 1 or settings.n_init < 1:
        raise ValueError("clustering n_clusters and n_init must be positive")
    if len(teams) < settings.n_clusters:
        raise ValueError(f"Need at least {settings.n_clusters} teams; found {len(teams)}")
    if filtered.empty:
        raise ValueError("No teams satisfy the score filter")
    all_result = teams.copy()
    filtered_result = filtered.copy()
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(_log_features(all_result))
    kmeans = KMeans(
        n_clusters=settings.n_clusters,
        random_state=settings.random_state,
        n_init=settings.n_init,
    )
    all_result["cluster"] = kmeans.fit_predict(all_scaled)
    filtered_result["cluster"] = kmeans.predict(scaler.transform(_log_features(filtered_result)))
    centers = np.expm1(scaler.inverse_transform(kmeans.cluster_centers_))
    summary = pd.DataFrame(
        {
            "cluster": np.arange(settings.n_clusters, dtype=int),
            "center_startup_seconds": centers[:, 0],
            "center_step_seconds": centers[:, 1],
        }
    )
    summary["all_team_count"] = summary["cluster"].map(all_result["cluster"].value_counts()).fillna(0).astype(int)
    summary["filtered_team_count"] = summary["cluster"].map(filtered_result["cluster"].value_counts()).fillna(0).astype(int)
    return all_result, filtered_result, summary


def main(config_path: Path = CONFIG_PATH) -> None:
    settings = load_settings(config_path)
    date, path = resolve_timing_csv(settings.input, settings.date)
    players = pd.read_csv(path, dtype={"episode_id": "string", "team_name": "string"})
    teams = aggregate_team_timings(players)
    filtered = filter_team_timings(teams, settings.score_filter)
    assign_timing_clusters(teams, filtered, settings.clustering)
    print(f"Prepared Replay Timing EDA for date={date}")


if __name__ == "__main__":
    main()
