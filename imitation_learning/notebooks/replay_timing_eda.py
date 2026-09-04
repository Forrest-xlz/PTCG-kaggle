"""Generate Replay Timing EDA figures and tables from extracted timing data."""
from __future__ import annotations

import re
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "cfg" / "replay_timing_eda.yaml"
DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})$")
SCORE_COLUMNS = {"min": "min_score", "max": "max_score", "avg": "avg_score"}
FEATURE_COLUMNS = ("startup_time_mean_seconds", "mean_step_time_seconds")
FIGURE_FILENAMES = (
    "all_teams_startup_distribution.png",
    "all_teams_action_distribution.png",
    "score_filtered_startup_distribution.png",
    "score_filtered_action_distribution.png",
    "all_teams_startup_vs_action.png",
    "score_filtered_startup_vs_action.png",
)
TABLE_FILENAMES = (
    "all_team_timings.csv",
    "score_filtered_team_timings.csv",
    "cluster_summary.csv",
)
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


def _add_figure_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.08, y=0.98, ha="left", va="top", fontsize=15, fontweight="semibold")
    fig.text(0.08, 0.91, subtitle, ha="left", va="top", color="#5B6472", fontsize=10)


def _save_histogram(
    frame: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    color: str,
    destination: Path,
) -> None:
    values = frame.loc[frame[column].gt(0), column]
    if values.empty:
        raise ValueError(f"No positive values available for {title}")
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        sns.histplot(values, bins=35, color=color, edgecolor="white", linewidth=0.6, ax=ax)
        ax.set_xscale("log")
        _add_figure_header(fig, title, f"Unique teams: {len(values):,} | Unit: seconds | Log-scaled x-axis")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Teams")
        sns.despine(ax=ax)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)


def _save_scatter(
    frame: pd.DataFrame,
    centers: pd.DataFrame,
    title: str,
    subtitle: str,
    destination: Path,
) -> None:
    plotted = frame.loc[
        frame["mean_step_time_seconds"].gt(0) & frame["startup_time_mean_seconds"].gt(0)
    ].copy()
    if plotted.empty:
        raise ValueError(f"No positive timing pairs available for {title}")
    fig, ax = plt.subplots(figsize=(10, 7))
    try:
        palette = sns.color_palette("tab10", n_colors=max(1, len(centers)))
        for cluster in centers["cluster"].astype(int):
            rows = plotted.loc[plotted["cluster"].eq(cluster)]
            if rows.empty:
                continue
            ax.scatter(
                rows["mean_step_time_seconds"],
                rows["startup_time_mean_seconds"],
                s=38,
                alpha=0.72,
                color=palette[cluster % len(palette)],
                label=f"Cluster {cluster}",
                edgecolors="white",
                linewidths=0.35,
            )
        ax.scatter(
            centers["center_step_seconds"],
            centers["center_startup_seconds"],
            s=190,
            marker="X",
            color="#111827",
            edgecolors="white",
            linewidths=0.9,
            label="Global centers",
            zorder=5,
        )
        for row in centers.itertuples(index=False):
            ax.annotate(
                str(row.cluster),
                (row.center_step_seconds, row.center_startup_seconds),
                xytext=(7, 6),
                textcoords="offset points",
                color="#111827",
                fontsize=10,
                weight="bold",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        _add_figure_header(fig, title, subtitle)
        ax.set_xlabel("Mean subsequent action time (seconds)")
        ax.set_ylabel("Mean startup time (seconds)")
        ax.legend(frameon=False, ncol=3, loc="best")
        sns.despine(ax=ax)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)


def _write_artifacts(
    staging: Path,
    all_teams: pd.DataFrame,
    filtered: pd.DataFrame,
    summary: pd.DataFrame,
    score_filter: ScoreFilter,
) -> None:
    (staging / "tables").mkdir(parents=True, exist_ok=True)
    filter_label = f"{score_filter.mode}_score >= {score_filter.threshold:g}"
    _save_histogram(all_teams, "startup_time_mean_seconds", "All Teams: Startup Time Distribution", "Mean startup time (seconds)", "#2F6BFF", staging / FIGURE_FILENAMES[0])
    _save_histogram(all_teams, "mean_step_time_seconds", "All Teams: Mean Subsequent Action Time Distribution", "Mean subsequent action time (seconds)", "#2F6BFF", staging / FIGURE_FILENAMES[1])
    _save_histogram(filtered, "startup_time_mean_seconds", f"Score-Filtered Teams: Startup Time Distribution ({filter_label})", "Mean startup time (seconds)", "#D97706", staging / FIGURE_FILENAMES[2])
    _save_histogram(filtered, "mean_step_time_seconds", f"Score-Filtered Teams: Mean Subsequent Action Time Distribution ({filter_label})", "Mean subsequent action time (seconds)", "#D97706", staging / FIGURE_FILENAMES[3])
    _save_scatter(all_teams, summary, "All Teams: Startup vs Mean Subsequent Action Time", f"Teams: {len(all_teams):,} | Global K-Means ({len(summary)} clusters) | Log-scaled axes", staging / FIGURE_FILENAMES[4])
    _save_scatter(filtered, summary, f"Score-Filtered Teams: Startup vs Mean Subsequent Action Time ({filter_label})", f"Teams: {len(filtered):,} | Assigned with global centers | Log-scaled axes", staging / FIGURE_FILENAMES[5])
    all_teams.sort_values(["cluster", "team_name"], kind="stable").to_csv(staging / "tables" / TABLE_FILENAMES[0], index=False, encoding="utf-8-sig")
    filtered_output = filtered.copy()
    filtered_output.insert(1, "score_mode", score_filter.mode)
    filtered_output.sort_values(["score_value", "team_name"], ascending=[False, True], kind="stable").to_csv(staging / "tables" / TABLE_FILENAMES[1], index=False, encoding="utf-8-sig")
    summary.to_csv(staging / "tables" / TABLE_FILENAMES[2], index=False, encoding="utf-8-sig")


def publish_artifacts(staging: Path, output: Path) -> None:
    sources = [staging / name for name in FIGURE_FILENAMES]
    sources.extend(staging / "tables" / name for name in TABLE_FILENAMES)
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing staged Replay Timing artifacts: {missing}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    for name in FIGURE_FILENAMES:
        os.replace(staging / name, output / name)
    for name in TABLE_FILENAMES:
        os.replace(staging / "tables" / name, output / "tables" / name)


def run(settings: AnalysisSettings) -> tuple[str, Path]:
    date, path = resolve_timing_csv(settings.input, settings.date)
    players = pd.read_csv(path, dtype={"episode_id": "string", "team_name": "string"})
    teams = aggregate_team_timings(players)
    filtered = filter_team_timings(teams, settings.score_filter)
    all_result, filtered_result, summary = assign_timing_clusters(teams, filtered, settings.clustering)
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="replay-timing-", dir=settings.output.parent) as temporary:
        staging = Path(temporary)
        _write_artifacts(staging, all_result, filtered_result, summary, settings.score_filter)
        publish_artifacts(staging, settings.output)
    return date, settings.output


def main(config_path: Path = CONFIG_PATH) -> None:
    settings = load_settings(config_path)
    date, output = run(settings)
    print(f"Saved Replay Timing EDA for date={date} to {output}")


if __name__ == "__main__":
    main()
