from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deck.analysis import CardCatalog
from deck.trend import (
    add_trend_archetypes,
    build_daily_metrics,
    build_matchups,
    build_team_flows,
    build_team_modal_archetypes,
    classify_trend_archetype,
    select_snapshot_dates,
)


def catalog() -> CardCatalog:
    return CardCatalog(
        names={
            1: "Filler",
            2: "Great Tusk",
            3: "Crustle",
            4: "Marnie's Grimmsnarl ex",
        },
        kinds={1: "Energy", 2: "Pokemon", 3: "Pokemon", 4: "Pokemon"},
    )


def deck(*markers: int) -> list[int]:
    return list(markers) + [1] * (60 - len(markers))


def player_rows() -> pd.DataFrame:
    records = [
        ("6.26", "a", 0, "A", "B", deck(4), 1.0, "win"),
        ("6.26", "a", 1, "B", "A", deck(2, 3), -1.0, "loss"),
        ("6.26", "b", 0, "A", "C", deck(2, 3), -1.0, "loss"),
        ("6.26", "b", 1, "C", "A", deck(4), 1.0, "win"),
        ("6.29", "c", 0, "A", "B", deck(4), 1.0, "win"),
        ("6.29", "c", 1, "B", "A", deck(1), -1.0, "loss"),
    ]
    frame = pd.DataFrame(
        records,
        columns=[
            "date",
            "episode_id",
            "player",
            "team_name",
            "opponent_team_name",
            "deck",
            "reward",
            "result",
        ],
    )
    frame["deck"] = frame["deck"].map(
        lambda value: json.dumps(value, separators=(",", ":"))
    )
    return frame


def test_rule_only_classifier_uses_other_without_fallback() -> None:
    assert classify_trend_archetype(deck(4), catalog()) == "Marnie Grimmsnarl"
    assert classify_trend_archetype(deck(2, 3), catalog()) == (
        "Great Tusk / Crustle"
    )
    assert classify_trend_archetype(deck(1), catalog()) == "Other"


def test_snapshot_dates_use_calendar_gaps() -> None:
    assert select_snapshot_dates(
        ["6.26", "6.27", "6.29", "7.1", "7.2"], 3
    ) == ["6.26", "6.29", "7.2"]
    assert select_snapshot_dates(
        ["6.26", "6.27", "6.29", "7.1"],
        1,
        start_date="6.27",
        end_date="7.1",
    ) == ["6.27", "6.29", "7.1"]
    with pytest.raises(ValueError, match="positive"):
        select_snapshot_dates(["6.26"], 0)


def test_metrics_modal_flows_and_matchups_reconcile() -> None:
    annotated = add_trend_archetypes(player_rows(), catalog())
    daily = build_daily_metrics(annotated, exclude_mirrors=True)
    assert daily.groupby("date")["share_percent"].sum().tolist() == pytest.approx(
        [100.0, 100.0]
    )

    modal = build_team_modal_archetypes(annotated)
    team_a_first = modal[
        (modal["date"] == "6.26") & (modal["team_name"] == "A")
    ].iloc[0]
    assert team_a_first["modal_archetype"] == "Great Tusk / Crustle"

    flows, retention = build_team_flows(modal, ["6.26", "6.29"])
    assert int(flows["teams"].sum()) == 2
    assert retention.iloc[0]["overlapping_teams"] == 2
    assert retention.iloc[0]["from_retention_rate"] == pytest.approx(2 / 3)
    assert retention.iloc[0]["to_retention_rate"] == pytest.approx(1.0)
    assert retention.iloc[0]["switched_teams"] == 1

    matchups = build_matchups(annotated, exclude_mirrors=True)
    assert int(matchups["games"].sum()) == len(annotated)
    assert (
        matchups["wins"] + matchups["losses"] + matchups["draws"]
    ).equals(matchups["games"])
