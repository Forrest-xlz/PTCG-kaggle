"""Pure helpers for replay-level Deck trend analysis."""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Iterable, Sequence

import pandas as pd

from deck.analysis import (
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
