"""Generate the three final Deck Trends EDA figures from fixed YAML."""
from __future__ import annotations

import io
import os
import shutil
import sys
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.deck_statistics import CardCatalog
from analysis.deck_trends import (
    ScoreFilter,
    add_trend_archetypes,
    build_daily_metrics,
    build_matchups,
    build_pooled_archetype_shares,
    build_team_flows,
    build_team_modal_archetypes,
    daily_share_visibility,
    filter_replays_by_score,
    parse_month_day,
    plot_archetype_sankey,
    select_snapshot_dates,
)

CONFIG_PATH = PROJECT_ROOT / "cfg" / "deck_trends_eda.yaml"
FIGURE_FILENAMES = {
    "line_chart": "archetype_share_and_win_rate.png",
    "sankey": "archetype_sankey.png",
    "matchup_matrix": "archetype_matchup_matrix.png",
}
TABLE_FILENAMES = ("line_chart_daily_metrics.csv", "matchup_matrix.csv")


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _chart_settings(raw: dict[str, Any], name: str, *, mirrors: bool) -> dict[str, Any]:
    cfg = dict(raw[name])
    cfg["interval_days"] = int(cfg["interval_days"])
    cfg["min_share_percent"] = float(cfg["min_share_percent"])
    if cfg["interval_days"] < 1 or not 0 <= cfg["min_share_percent"] <= 100:
        raise ValueError(f"invalid {name} interval/share setting")
    cfg["score_filter"] = ScoreFilter(cfg["score_mode"], cfg["score_threshold"])
    if mirrors and type(cfg["exclude_mirror_matches"]) is not bool:
        raise ValueError(f"{name}.exclude_mirror_matches must be boolean")
    return cfg


def _load_rows(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*.players.csv.gz"), key=lambda p: parse_month_day(p.name.split(".players.csv.gz")[0]))
    if not paths:
        raise FileNotFoundError(f"No trend shards found in {data_dir}")
    rows = pd.concat([pd.read_csv(p, dtype={"date": str, "episode_id": str}) for p in paths], ignore_index=True)
    if not rows.groupby(["date", "episode_id"]).size().eq(2).all():
        raise ValueError("Every replay must contain exactly two player rows")
    return rows


def _load_scores(replay_dir: Path, dates: set[str]) -> pd.DataFrame:
    frames = []
    for date in sorted(dates, key=parse_month_day):
        path = replay_dir / f"{date}.zip"
        with zipfile.ZipFile(path) as archive:
            matches = [n for n in archive.namelist() if Path(n).name == "manifest.csv"]
            if len(matches) != 1:
                raise FileNotFoundError(f"{path} must contain one manifest.csv")
            with archive.open(matches[0]) as raw:
                frame = pd.read_csv(io.TextIOWrapper(raw, encoding="utf-8"), dtype={"episode_id": str}, usecols=["episode_id", "avg_score", "min_score", "sum_score"])
        frame.insert(0, "date", date)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _prepare(rows: pd.DataFrame, catalog: CardCatalog, cfg: dict[str, Any], scores: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    dates = select_snapshot_dates(rows["date"].unique(), cfg["interval_days"], start_date=cfg.get("start_date"), end_date=cfg.get("end_date"))
    if not dates:
        raise ValueError("No dates remain for chart")
    selected = rows[rows["date"].isin(dates)].copy()
    selected = filter_replays_by_score(selected, scores, cfg["score_filter"])
    if selected.empty:
        raise ValueError("No replays remain after score filtering")
    return add_trend_archetypes(selected, catalog), dates


def _save_line(rows: pd.DataFrame, dates: list[str], cfg: dict[str, Any], figure_path: Path, table_path: Path) -> None:
    metrics = build_daily_metrics(rows, exclude_mirrors=cfg["exclude_mirror_matches"]).rename(columns={"non_mirror_win_rate": "win_rate", "non_mirror_games": "evaluated_games"})
    visible_names = sorted(set(metrics.loc[daily_share_visibility(metrics, cfg["min_share_percent"]), "archetype"]))
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    positions = np.arange(len(dates))
    colors = dict(zip(visible_names, sns.color_palette("tab20", n_colors=len(visible_names))))
    for name in visible_names:
        series = metrics[metrics["archetype"] == name].set_index("date").reindex(dates)
        visible = series["share_percent"].ge(cfg["min_share_percent"])
        axes[0].plot(positions, series["share_percent"].where(visible), marker="o", label=name, color=colors[name])
        axes[1].plot(positions, series["win_rate"].where(visible), marker="o", color=colors[name])
    axes[0].set(title="Archetype Share Across Selected Snapshots", ylabel="Share (%)")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1)
    axes[1].set(title="Archetype Win Rate Across Selected Snapshots", ylabel="Win rate", xlabel="Snapshot",
                xticks=positions, xticklabels=dates)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    metrics.to_csv(table_path, index=False, encoding="utf-8-sig")


def _save_sankey(rows: pd.DataFrame, dates: list[str], cfg: dict[str, Any], figure_path: Path) -> None:
    metrics = build_daily_metrics(rows)
    shares = metrics.set_index(["date", "archetype"])["share_percent"].to_dict()
    display_rows = rows[["date", "archetype"]].copy()
    display_rows["display_archetype"] = display_rows.apply(lambda r: r["archetype"] if shares.get((r["date"], r["archetype"]), 0) >= cfg["min_share_percent"] else "Other", axis=1)
    display_shares = display_rows.groupby(["date", "display_archetype"]).size().rename("uses").reset_index()
    display_shares["share_percent"] = 100 * display_shares["uses"] / display_shares.groupby("date")["uses"].transform("sum")
    modal = build_team_modal_archetypes(rows)
    modal["modal_archetype"] = modal.apply(lambda r: r["modal_archetype"] if shares.get((r["date"], r["modal_archetype"]), 0) >= cfg["min_share_percent"] else "Other", axis=1)
    flows, _ = build_team_flows(modal, dates)
    fig = plot_archetype_sankey(display_shares, flows, dates)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_matrix(rows: pd.DataFrame, cfg: dict[str, Any], figure_path: Path, table_path: Path) -> None:
    matchups = build_matchups(rows, exclude_mirrors=cfg["exclude_mirror_matches"])
    shares = build_pooled_archetype_shares(build_daily_metrics(rows))
    names = sorted(shares.loc[shares["share_percent"] >= cfg["min_share_percent"], "archetype"])
    table = matchups[matchups["row_archetype"].isin(names) & matchups["column_archetype"].isin(names)].copy()
    if table.empty:
        raise ValueError("No matchup remains after share filtering")
    rates = table.pivot(index="row_archetype", columns="column_archetype", values="win_rate").reindex(index=names, columns=names)
    games = table.pivot(index="row_archetype", columns="column_archetype", values="games").reindex(index=names, columns=names)
    labels = rates.copy().astype(object)
    for row in names:
        for col in names:
            labels.loc[row, col] = "" if pd.isna(rates.loc[row, col]) else f"{rates.loc[row, col]:.0%}\n(n={int(games.loc[row, col]):,})"
    fig, ax = plt.subplots(figsize=(max(10, len(names) * .85), max(8, len(names) * .7)))
    sns.heatmap(rates, annot=labels, fmt="", cmap="RdYlGn", vmin=0, vmax=1, center=.5, linewidths=.5, ax=ax)
    ax.set(title="Pooled Archetype Matchups (Row Beats Column)", xlabel="Opponent archetype", ylabel="Player archetype")
    fig.tight_layout(); fig.savefig(figure_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    table.to_csv(table_path, index=False, encoding="utf-8-sig")


def _publish_artifacts(staging: Path, output: Path) -> None:
    sources = [staging / name for name in FIGURE_FILENAMES.values()]
    sources.extend(staging / "tables" / name for name in TABLE_FILENAMES)
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing staged Deck Trends artifacts: {missing}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    for name in FIGURE_FILENAMES.values():
        os.replace(staging / name, output / name)
    for name in TABLE_FILENAMES:
        os.replace(staging / "tables" / name, output / "tables" / name)


@contextmanager
def _staging_directory(output: Path):
    staging = output.parent / f".{output.name}-staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(config_path: Path = CONFIG_PATH) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))["analysis"]
    configs = {"line": _chart_settings(raw, "line_chart", mirrors=True), "sankey": _chart_settings(raw, "sankey", mirrors=False), "matrix": _chart_settings(raw, "matchup_matrix", mirrors=True)}
    rows = _load_rows(_path(raw["input"]))
    catalog = CardCatalog.from_csv(_path(raw["card_table"]))
    required_dates: set[str] = set()
    for cfg in configs.values():
        if cfg["score_filter"].mode != "all":
            required_dates.update(select_snapshot_dates(
                rows["date"].unique(), cfg["interval_days"],
                start_date=cfg.get("start_date"), end_date=cfg.get("end_date"),
            ))
    scores = _load_scores(_path(raw["replay_episodes"]), required_dates) if required_dates else None
    output = _path(raw["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with _staging_directory(output) as staging:
        (staging / "tables").mkdir(parents=True, exist_ok=True)
        line_rows, line_dates = _prepare(rows, catalog, configs["line"], scores)
        _save_line(line_rows, line_dates, configs["line"], staging / FIGURE_FILENAMES["line_chart"], staging / "tables" / TABLE_FILENAMES[0])
        sankey_rows, sankey_dates = _prepare(rows, catalog, configs["sankey"], scores)
        _save_sankey(sankey_rows, sankey_dates, configs["sankey"], staging / FIGURE_FILENAMES["sankey"])
        matrix_rows, _ = _prepare(rows, catalog, configs["matrix"], scores)
        _save_matrix(matrix_rows, configs["matrix"], staging / FIGURE_FILENAMES["matchup_matrix"], staging / "tables" / TABLE_FILENAMES[1])
        _publish_artifacts(staging, output)
    print(f"Saved Deck Trends EDA to {output}")


if __name__ == "__main__":
    main()
