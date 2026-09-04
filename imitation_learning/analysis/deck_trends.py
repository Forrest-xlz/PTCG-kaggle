"""Pure helpers for replay-level Deck trend analysis."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Sequence

import pandas as pd

from analysis.deck_statistics import (
    ARCHETYPE_RULES,
    CardCatalog,
    normalize_card_name,
)


PLAYER_ROW_COLUMNS = {
    "date",
    "episode_id",
    "player",
    "team_name",
    "opponent_team_name",
    "deck",
    "reward",
    "result",
}


@dataclass(frozen=True)
class ScoreFilter:
    mode: str = "all"
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"all", "min", "max", "avg"}:
            raise ValueError("score mode must be all, min, max, or avg")
        if self.mode == "all":
            if self.threshold is not None:
                raise ValueError("score_threshold must be null for score_mode=all")
            return
        if isinstance(self.threshold, bool) or self.threshold is None:
            raise ValueError(f"score_threshold is required for score_mode={self.mode}")
        threshold = float(self.threshold)
        if not math.isfinite(threshold):
            raise ValueError("score_threshold must be finite")
        object.__setattr__(self, "threshold", threshold)


def filter_replays_by_score(
    rows: pd.DataFrame,
    scores: pd.DataFrame | None,
    score_filter: ScoreFilter,
) -> pd.DataFrame:
    """Filter complete replay pairs using one manifest-level score measure."""
    if score_filter.mode == "all":
        return rows.copy()
    if scores is None:
        raise ValueError("manifest scores are required for score filtering")
    required = {"date", "episode_id", "avg_score", "min_score", "sum_score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"manifest scores are missing columns: {sorted(missing)}")
    values = scores[list(required)].copy()
    values["date"] = values["date"].astype(str)
    values["episode_id"] = values["episode_id"].astype(str)
    if values.duplicated(["date", "episode_id"]).any():
        raise ValueError("manifest contains duplicate date/episode_id rows")
    for column in ("avg_score", "min_score", "sum_score"):
        values[column] = pd.to_numeric(values[column], errors="raise")
        if not values[column].map(math.isfinite).all():
            raise ValueError("manifest scores must be finite")
    score = {
        "min": values["min_score"],
        "max": values["sum_score"] - values["min_score"],
        "avg": values["avg_score"],
    }[score_filter.mode]
    allowed = values.loc[
        score.ge(float(score_filter.threshold)), ["date", "episode_id"]
    ]
    keyed = rows.copy()
    keyed["date"] = keyed["date"].astype(str)
    keyed["episode_id"] = keyed["episode_id"].astype(str)
    return keyed.merge(allowed, on=["date", "episode_id"], how="inner")

SANKEY_PALETTE = (
    "#8e44ad",
    "#d16ba5",
    "#c77c35",
    "#4169d8",
    "#59b3c3",
    "#7692ad",
    "#4f9d7a",
    "#5da349",
    "#b09b3b",
    "#7aa6d8",
    "#557f5f",
    "#b6a38a",
    "#d4bd24",
    "#ef8a62",
    "#9c6ade",
    "#17becf",
    "#bcbd22",
    "#e377c2",
    "#8c564b",
    "#1f77b4",
)


@dataclass(frozen=True, slots=True)
class SankeyNode:
    date: str
    archetype: str
    share_percent: float
    uses: int
    rank: int
    x: float
    bottom: float
    top: float
    color: str

    @property
    def key(self) -> tuple[str, str]:
        return self.date, self.archetype

    @property
    def height(self) -> float:
        return self.top - self.bottom


@dataclass(frozen=True, slots=True)
class SankeyRibbon:
    source: tuple[str, str]
    target: tuple[str, str]
    teams: int
    source_bottom: float
    source_top: float
    target_bottom: float
    target_top: float
    color: str


@dataclass(frozen=True, slots=True)
class SankeyLayout:
    dates: tuple[str, ...]
    nodes: tuple[SankeyNode, ...]
    ribbons: tuple[SankeyRibbon, ...]
    color_by_archetype: dict[str, str]
    games_by_date: dict[str, int]
    node_width: float


def _parse_deck(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("deck contains invalid JSON") from exc
    if not isinstance(value, list) or len(value) != 60:
        length = len(value) if isinstance(value, list) else "non-list"
        raise ValueError(f"deck must contain exactly 60 cards, found {length}")
    if any(type(card_id) is not int or card_id < 0 for card_id in value):
        raise ValueError("deck must contain non-negative integer Card IDs")
    return value


def classify_trend_archetype(
    deck: Iterable[int],
    catalog: CardCatalog,
) -> str:
    """Classify with curated rules only; never use a main-Pokemon fallback."""
    cards = _parse_deck(list(deck))
    present = {
        normalize_card_name(catalog.name(card_id)) for card_id in cards
    }
    for rule in ARCHETYPE_RULES:
        required_all = {
            normalize_card_name(name) for name in rule.get("all", [])
        }
        required_any = {
            normalize_card_name(name) for name in rule.get("any", [])
        }
        if required_all and not required_all.issubset(present):
            continue
        if required_any and not (required_any & present):
            continue
        return str(rule["name"])
    return "Other"


def parse_month_day(value: str) -> date:
    """Parse an unpadded month.day label on a leap-safe reference year."""
    text = str(value).strip()
    parts = text.split(".")
    if len(parts) != 2:
        raise ValueError(f"invalid month.day date: {value}")
    try:
        month, day = map(int, parts)
        return date(2000, month, day)
    except ValueError as exc:
        raise ValueError(f"invalid month.day date: {value}") from exc


def select_snapshot_dates(
    values: Iterable[str],
    interval_days: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    """Select the earliest available snapshots separated by calendar days."""
    if type(interval_days) is not int or interval_days < 1:
        raise ValueError("interval_days must be a positive integer")
    dated_values = []
    for value in values:
        parsed = parse_month_day(value)
        dated_values.append((parsed, f"{parsed.month}.{parsed.day}"))
    dated = sorted(set(dated_values))
    start = parse_month_day(start_date) if start_date is not None else None
    end = parse_month_day(end_date) if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must not be after end_date")
    eligible = [
        (parsed, label)
        for parsed, label in dated
        if (start is None or parsed >= start) and (end is None or parsed <= end)
    ]
    if not eligible:
        return []
    selected = [eligible[0]]
    for candidate in eligible[1:]:
        if (candidate[0] - selected[-1][0]).days >= interval_days:
            selected.append(candidate)
    return [label for _, label in selected]


def _validate_player_rows(rows: pd.DataFrame) -> pd.DataFrame:
    missing = PLAYER_ROW_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError(f"player rows are missing columns: {sorted(missing)}")
    result = rows.copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("player rows are empty")
    result["date"] = result["date"].astype(str)
    result["episode_id"] = result["episode_id"].astype(str)
    result["team_name"] = result["team_name"].astype(str).str.strip()
    result["opponent_team_name"] = (
        result["opponent_team_name"].astype(str).str.strip()
    )
    invalid_names = {"", "?", "nan", "None"}
    if result["team_name"].isin(invalid_names).any():
        raise ValueError("team_name must not be empty or unknown")
    if result["opponent_team_name"].isin(invalid_names).any():
        raise ValueError("opponent_team_name must not be empty or unknown")
    if not set(result["result"]).issubset({"win", "loss", "draw"}):
        raise ValueError("result must contain only win, loss, or draw")
    grouped = result.groupby(
        ["date", "episode_id"], sort=False, observed=True
    )
    sizes = grouped.size()
    if not sizes.eq(2).all():
        raise ValueError("each replay must contain exactly two player rows")
    for key, group in grouped:
        if set(group["player"].astype(int)) != {0, 1}:
            raise ValueError(f"replay {key} must contain player indices 0 and 1")
    return result


def add_trend_archetypes(
    rows: pd.DataFrame,
    catalog: CardCatalog,
) -> pd.DataFrame:
    """Classify unique exact Decks once and attach opponent archetypes."""
    result = _validate_player_rows(rows)
    parsed = result["deck"].map(_parse_deck)
    canonical = parsed.map(lambda cards: tuple(sorted(cards)))
    archetype_by_deck = {
        deck: classify_trend_archetype(deck, catalog)
        for deck in canonical.drop_duplicates()
    }
    result["archetype"] = canonical.map(archetype_by_deck)
    opponent = pd.Series(index=result.index, dtype=object)
    for _, group in result.groupby(
        ["date", "episode_id"], sort=False, observed=True
    ):
        first, second = group.index.tolist()
        opponent.at[first] = result.at[second, "archetype"]
        opponent.at[second] = result.at[first, "archetype"]
    result["opponent_archetype"] = opponent
    return result


def build_daily_metrics(
    rows: pd.DataFrame,
    exclude_mirrors: bool = True,
) -> pd.DataFrame:
    """Compute usage share and field win rate for each date/archetype."""
    required = {"date", "archetype", "opponent_archetype", "result"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"annotated rows are missing columns: {sorted(missing)}")
    usage = (
        rows.groupby(
            ["date", "archetype"], sort=False, observed=True
        )
        .size()
        .rename("uses")
        .reset_index()
    )
    totals = usage.groupby("date")["uses"].transform("sum")
    usage["share_percent"] = 100.0 * usage["uses"] / totals

    field = rows
    if exclude_mirrors:
        field = field[field["archetype"] != field["opponent_archetype"]]
    counts = (
        field.groupby(
            ["date", "archetype", "result"], sort=False, observed=True
        )
        .size()
        .unstack(fill_value=0)
    )
    for column in ("win", "loss", "draw"):
        if column not in counts:
            counts[column] = 0
    counts = counts[["win", "loss", "draw"]].rename(
        columns={"win": "wins", "loss": "losses", "draw": "draws"}
    )
    counts["non_mirror_games"] = counts.sum(axis=1)
    denominator = counts["non_mirror_games"].where(
        counts["non_mirror_games"].ne(0)
    )
    counts["non_mirror_win_rate"] = counts["wins"].div(denominator)
    result = usage.merge(
        counts.reset_index(), on=["date", "archetype"], how="left"
    )
    for column in ("wins", "losses", "draws", "non_mirror_games"):
        result[column] = result[column].fillna(0).astype(int)
    return result.sort_values(
        ["date", "uses", "archetype"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _validate_share_threshold(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("share threshold must be between 0 and 100")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("share threshold must be between 0 and 100") from exc
    if pd.isna(threshold) or not 0 <= threshold <= 100:
        raise ValueError("share threshold must be between 0 and 100")
    return threshold


def daily_share_visibility(
    rows: pd.DataFrame,
    min_share_percent: float,
) -> pd.Series:
    """Return which archetype/date rows meet the configured daily share."""
    if "share_percent" not in rows:
        raise ValueError("daily metrics are missing share_percent")
    threshold = _validate_share_threshold(min_share_percent)
    shares = pd.to_numeric(rows["share_percent"], errors="raise")
    if shares.isna().any() or not shares.between(0, 100).all():
        raise ValueError("daily share_percent values must be between 0 and 100")
    return shares.ge(threshold)


def build_pooled_archetype_shares(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate uses-weighted archetype share across selected snapshots."""
    required = {"archetype", "uses"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"daily metrics are missing columns: {sorted(missing)}")
    values = rows[["archetype", "uses"]].copy()
    values["archetype"] = values["archetype"].astype(str)
    values["uses"] = pd.to_numeric(values["uses"], errors="raise")
    if values["uses"].isna().any() or (values["uses"] < 0).any():
        raise ValueError("daily metric uses must be non-negative")
    pooled = (
        values.groupby("archetype", as_index=False, observed=True)["uses"]
        .sum()
    )
    total_uses = pooled["uses"].sum()
    if total_uses <= 0:
        raise ValueError("pooled archetype uses must be positive")
    pooled["share_percent"] = 100.0 * pooled["uses"] / total_uses
    return pooled.sort_values(
        ["uses", "archetype"], ascending=[False, True]
    ).reset_index(drop=True)


def build_team_modal_archetypes(rows: pd.DataFrame) -> pd.DataFrame:
    """Choose each team's most-used archetype per date deterministically."""
    required = {"date", "team_name", "archetype"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"annotated rows are missing columns: {sorted(missing)}")
    counts = (
        rows.groupby(
            ["date", "team_name", "archetype"],
            sort=False,
            observed=True,
        )
        .size()
        .rename("games")
        .reset_index()
        .sort_values(
            ["date", "team_name", "games", "archetype"],
            ascending=[True, True, False, True],
        )
    )
    modal = counts.drop_duplicates(["date", "team_name"], keep="first")
    return modal.rename(columns={"archetype": "modal_archetype"}).reset_index(drop=True)


def build_team_flows(
    modal: pd.DataFrame,
    dates: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count modal-archetype transitions and adjacent-snapshot retention."""
    required = {"date", "team_name", "modal_archetype"}
    missing = required - set(modal.columns)
    if missing:
        raise ValueError(f"modal rows are missing columns: {sorted(missing)}")
    flow_parts: list[pd.DataFrame] = []
    retention_rows: list[dict[str, Any]] = []
    for from_date, to_date in zip(dates, dates[1:]):
        first = modal[modal["date"] == from_date][
            ["team_name", "modal_archetype"]
        ].rename(columns={"modal_archetype": "source_archetype"})
        second = modal[modal["date"] == to_date][
            ["team_name", "modal_archetype"]
        ].rename(columns={"modal_archetype": "target_archetype"})
        overlap = first.merge(second, on="team_name", how="inner")
        if not overlap.empty:
            flows = (
                overlap.groupby(["source_archetype", "target_archetype"])
                .size()
                .rename("teams")
                .reset_index()
            )
            flows.insert(0, "to_date", to_date)
            flows.insert(0, "from_date", from_date)
            flow_parts.append(flows)
        unchanged = int(
            overlap["source_archetype"].eq(overlap["target_archetype"]).sum()
        )
        overlapping = len(overlap)
        switched = overlapping - unchanged
        retention_rows.append(
            {
                "from_date": from_date,
                "to_date": to_date,
                "from_teams": len(first),
                "to_teams": len(second),
                "overlapping_teams": overlapping,
                "from_retention_rate": (
                    overlapping / len(first) if len(first) else pd.NA
                ),
                "to_retention_rate": (
                    overlapping / len(second) if len(second) else pd.NA
                ),
                "unchanged_teams": unchanged,
                "switched_teams": switched,
                "switch_rate": switched / overlapping if overlapping else pd.NA,
            }
        )
    flow_columns = [
        "from_date",
        "to_date",
        "source_archetype",
        "target_archetype",
        "teams",
    ]
    flows = (
        pd.concat(flow_parts, ignore_index=True)
        if flow_parts
        else pd.DataFrame(columns=flow_columns)
    )
    return flows, pd.DataFrame(retention_rows)


def build_matchups(
    rows: pd.DataFrame,
    exclude_mirrors: bool = True,
) -> pd.DataFrame:
    """Build a long-form row-archetype versus column-archetype table."""
    required = {"archetype", "opponent_archetype", "result"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"annotated rows are missing columns: {sorted(missing)}")
    field = rows
    if exclude_mirrors:
        field = field[field["archetype"] != field["opponent_archetype"]]
    counts = (
        field.groupby(["archetype", "opponent_archetype", "result"])
        .size()
        .unstack(fill_value=0)
    )
    for column in ("win", "loss", "draw"):
        if column not in counts:
            counts[column] = 0
    counts = counts[["win", "loss", "draw"]].rename(
        columns={"win": "wins", "loss": "losses", "draw": "draws"}
    )
    counts["games"] = counts.sum(axis=1)
    counts["win_rate"] = counts["wins"] / counts["games"]
    result = counts.reset_index().rename(
        columns={
            "archetype": "row_archetype",
            "opponent_archetype": "column_archetype",
        }
    )
    return result[
        [
            "row_archetype",
            "column_archetype",
            "games",
            "wins",
            "losses",
            "draws",
            "win_rate",
        ]
    ].sort_values(["row_archetype", "column_archetype"]).reset_index(drop=True)


def _sankey_colors(shares: pd.DataFrame) -> dict[str, str]:
    totals = (
        shares.groupby("display_archetype", observed=True)["uses"]
        .sum()
        .sort_values(ascending=False)
    )
    ordered = sorted(
        (str(name) for name in totals.index if str(name) != "Other"),
        key=lambda name: (-int(totals.loc[name]), name),
    )
    colors = {
        name: SANKEY_PALETTE[index % len(SANKEY_PALETTE)]
        for index, name in enumerate(ordered)
    }
    if "Other" in totals.index:
        colors["Other"] = "#9e9e9e"
    return colors


def build_sankey_layout(
    display_shares: pd.DataFrame,
    flows: pd.DataFrame,
    snapshot_dates: Sequence[str],
    *,
    bottom: float = 0.08,
    top: float = 0.92,
    gap: float = 0.006,
    node_width: float = 0.012,
) -> SankeyLayout:
    """Lay out share-sized nodes and globally scaled team-flow ribbons."""
    share_columns = {"date", "display_archetype", "share_percent", "uses"}
    missing = share_columns - set(display_shares.columns)
    if missing:
        raise ValueError(f"display shares are missing columns: {sorted(missing)}")
    flow_columns = {
        "from_date",
        "to_date",
        "source_archetype",
        "target_archetype",
        "teams",
    }
    missing = flow_columns - set(flows.columns)
    if missing:
        raise ValueError(f"flows are missing columns: {sorted(missing)}")
    dates = tuple(str(value) for value in snapshot_dates)
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("snapshot_dates must contain unique date labels")
    if not 0 <= bottom < top <= 1:
        raise ValueError("Sankey vertical bounds must satisfy 0 <= bottom < top <= 1")
    if gap < 0 or node_width <= 0:
        raise ValueError("Sankey gap must be non-negative and node width positive")

    shares = display_shares.copy()
    shares["date"] = shares["date"].astype(str)
    shares["display_archetype"] = shares["display_archetype"].astype(str)
    shares = shares[shares["date"].isin(dates)]
    shares["share_percent"] = pd.to_numeric(
        shares["share_percent"], errors="raise"
    )
    shares["uses"] = pd.to_numeric(shares["uses"], errors="raise")
    shares = (
        shares.groupby(
            ["date", "display_archetype"], as_index=False, observed=True
        )[["share_percent", "uses"]]
        .sum()
    )
    if (shares["share_percent"] <= 0).any() or (shares["uses"] < 0).any():
        raise ValueError("Sankey shares must be positive and uses non-negative")
    daily_totals = shares.groupby("date", observed=True)["share_percent"].sum()
    missing_dates = [date_label for date_label in dates if date_label not in daily_totals]
    if missing_dates:
        raise ValueError(f"display shares are missing dates: {missing_dates}")
    invalid_totals = daily_totals[~daily_totals.between(99.999, 100.001)]
    if not invalid_totals.empty:
        raise ValueError(
            "display shares must sum to 100% for every date: "
            f"{invalid_totals.to_dict()}"
        )

    colors = _sankey_colors(shares)
    x_by_date = {
        date_label: (
            0.5
            if len(dates) == 1
            else 0.03 + 0.94 * index / (len(dates) - 1)
        )
        for index, date_label in enumerate(dates)
    }
    nodes: list[SankeyNode] = []
    for date_label in dates:
        day = shares[shares["date"] == date_label].sort_values(
            ["share_percent", "display_archetype"],
            ascending=[False, True],
        )
        usable_height = top - bottom - gap * (len(day) - 1)
        if usable_height <= 0:
            raise ValueError(f"too many Sankey nodes for date {date_label}")
        cursor = top
        for rank, row in enumerate(day.itertuples(index=False)):
            height = usable_height * float(row.share_percent) / 100.0
            node_bottom = cursor - height
            archetype = str(row.display_archetype)
            nodes.append(
                SankeyNode(
                    date=date_label,
                    archetype=archetype,
                    share_percent=float(row.share_percent),
                    uses=int(row.uses),
                    rank=rank,
                    x=x_by_date[date_label],
                    bottom=node_bottom,
                    top=cursor,
                    color=colors[archetype],
                )
            )
            cursor = node_bottom - gap

    node_by_key = {node.key: node for node in nodes}
    date_pairs = set(zip(dates, dates[1:]))
    prepared_flows = flows.copy()
    for column in (
        "from_date",
        "to_date",
        "source_archetype",
        "target_archetype",
    ):
        prepared_flows[column] = prepared_flows[column].astype(str)
    prepared_flows["teams"] = pd.to_numeric(
        prepared_flows["teams"], errors="raise"
    )
    if (prepared_flows["teams"] < 0).any():
        raise ValueError("Sankey flow team counts must be non-negative")
    prepared_flows = prepared_flows[
        prepared_flows.apply(
            lambda row: (row["from_date"], row["to_date"]) in date_pairs,
            axis=1,
        )
    ]
    prepared_flows = (
        prepared_flows.groupby(
            [
                "from_date",
                "to_date",
                "source_archetype",
                "target_archetype",
            ],
            as_index=False,
            observed=True,
        )["teams"]
        .sum()
    )
    records = []
    for row in prepared_flows.itertuples(index=False):
        source = (row.from_date, row.source_archetype)
        target = (row.to_date, row.target_archetype)
        teams = int(row.teams)
        if source in node_by_key and target in node_by_key and teams > 0:
            records.append({"source": source, "target": target, "teams": teams})

    outgoing: dict[tuple[str, str], int] = {}
    incoming: dict[tuple[str, str], int] = {}
    for record in records:
        outgoing[record["source"]] = outgoing.get(record["source"], 0) + record[
            "teams"
        ]
        incoming[record["target"]] = incoming.get(record["target"], 0) + record[
            "teams"
        ]
    scale_candidates = []
    for key, node in node_by_key.items():
        teams = max(outgoing.get(key, 0), incoming.get(key, 0))
        if teams:
            scale_candidates.append(node.height / teams)
    ribbon_scale = min(scale_candidates) * 0.94 if scale_candidates else 0.0

    source_bounds: dict[int, tuple[float, float]] = {}
    target_bounds: dict[int, tuple[float, float]] = {}
    for key, node in node_by_key.items():
        source_indices = sorted(
            (index for index, record in enumerate(records) if record["source"] == key),
            key=lambda index: (
                node_by_key[records[index]["target"]].rank,
                records[index]["target"][1],
            ),
        )
        source_total = sum(records[index]["teams"] for index in source_indices)
        cursor = node.top - (node.height - source_total * ribbon_scale) / 2
        for index in source_indices:
            width = records[index]["teams"] * ribbon_scale
            source_bounds[index] = (cursor - width, cursor)
            cursor -= width

        target_indices = sorted(
            (index for index, record in enumerate(records) if record["target"] == key),
            key=lambda index: (
                node_by_key[records[index]["source"]].rank,
                records[index]["source"][1],
            ),
        )
        target_total = sum(records[index]["teams"] for index in target_indices)
        cursor = node.top - (node.height - target_total * ribbon_scale) / 2
        for index in target_indices:
            width = records[index]["teams"] * ribbon_scale
            target_bounds[index] = (cursor - width, cursor)
            cursor -= width

    ribbons = tuple(
        SankeyRibbon(
            source=record["source"],
            target=record["target"],
            teams=record["teams"],
            source_bottom=source_bounds[index][0],
            source_top=source_bounds[index][1],
            target_bottom=target_bounds[index][0],
            target_top=target_bounds[index][1],
            color=node_by_key[record["source"]].color,
        )
        for index, record in enumerate(records)
    )
    games_by_date = {
        date_label: int(round(shares.loc[shares["date"] == date_label, "uses"].sum() / 2))
        for date_label in dates
    }
    return SankeyLayout(
        dates=dates,
        nodes=tuple(nodes),
        ribbons=ribbons,
        color_by_archetype=colors,
        games_by_date=games_by_date,
        node_width=node_width,
    )


def _display_date(value: str) -> str:
    parsed = parse_month_day(value)
    return f"{parsed.month:02d}/{parsed.day:02d}"


def plot_archetype_sankey(
    display_shares: pd.DataFrame,
    flows: pd.DataFrame,
    snapshot_dates: Sequence[str],
):
    """Render a compact, share-faithful static archetype Sankey figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch, Patch, Rectangle
    from matplotlib.path import Path

    layout = build_sankey_layout(display_shares, flows, snapshot_dates)
    node_by_key = {node.key: node for node in layout.nodes}
    figure_width = max(14.0, 1.15 * len(layout.dates))
    figure, axis = plt.subplots(figsize=(figure_width, 8.0))

    for ribbon in sorted(layout.ribbons, key=lambda value: value.teams, reverse=True):
        source = node_by_key[ribbon.source]
        target = node_by_key[ribbon.target]
        source_x = source.x + layout.node_width / 2
        target_x = target.x - layout.node_width / 2
        control_1 = source_x + 0.42 * (target_x - source_x)
        control_2 = source_x + 0.58 * (target_x - source_x)
        vertices = [
            (source_x, ribbon.source_bottom),
            (control_1, ribbon.source_bottom),
            (control_2, ribbon.target_bottom),
            (target_x, ribbon.target_bottom),
            (target_x, ribbon.target_top),
            (control_2, ribbon.target_top),
            (control_1, ribbon.source_top),
            (source_x, ribbon.source_top),
            (source_x, ribbon.source_bottom),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        axis.add_patch(
            PathPatch(
                Path(vertices, codes),
                facecolor=to_rgba(ribbon.color, 0.30),
                edgecolor="none",
                zorder=1,
            )
        )

    for node in layout.nodes:
        axis.add_patch(
            Rectangle(
                (node.x - layout.node_width / 2, node.bottom),
                layout.node_width,
                node.height,
                facecolor=node.color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        )
        axis.text(
            node.x + layout.node_width * 0.7,
            (node.bottom + node.top) / 2,
            f"{node.share_percent:.1f}%",
            ha="left",
            va="center",
            fontsize=6.5,
            color="#27364d",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.4},
            zorder=4,
        )

    for date_label in layout.dates:
        x = next(node.x for node in layout.nodes if node.date == date_label)
        axis.text(
            x,
            0.965,
            _display_date(date_label),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#4b5563",
        )
        axis.text(
            x,
            0.035,
            f"{layout.games_by_date[date_label]:,}",
            ha="center",
            va="top",
            fontsize=6.5,
            color="#7a7a7a",
        )

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=archetype)
        for archetype, color in layout.color_by_archetype.items()
    ]
    if legend_handles:
        axis.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.075),
            ncol=min(8, max(1, len(legend_handles))),
            frameon=False,
            fontsize=7,
            columnspacing=1.2,
            handlelength=1.8,
        )
    total_games = sum(layout.games_by_date.values())
    first, last = _display_date(layout.dates[0]), _display_date(layout.dates[-1])
    axis.set_title(
        "PTCG AI Battle - top-band meta evolution "
        f"({first}-{last}, all {total_games:,} games; "
        "ribbons = team deck switches)",
        fontsize=13,
        pad=18,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.axis("off")
    figure.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.17)
    return figure
