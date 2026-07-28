from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_deck_extract_yaml_is_hierarchical() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "cfg" / "deck_extract.yaml").read_text(
            encoding="utf-8"
        )
    )
    extract = config["extract"]
    assert extract["input"]
    assert extract["output"]
    assert extract["workers"] >= 1
    assert extract["limit_members"] is None
    assert extract["force"] is False


def test_train_yaml_uses_validation_and_step_configuration() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "cfg" / "train.yaml").read_text(encoding="utf-8")
    )
    train = config["train"]
    assert "shuffle_mode" not in train
    assert "warmup_ratio" not in train
    assert train["warmup_steps"] >= 0
    assert train["validation_ratio"] == pytest.approx(0.05)
    assert train["expert_validation_ratio"] == pytest.approx(0.05)
    assert train["replay_episodes"]
    isolation = train["isolation_validation"]
    assert isolation["deck_data"] == "data/deck"
    assert set(isolation["selections"]) == {
        "deck_isolation",
        "archetype_isolation",
        "top_deck_archetype_isolation",
    }
    assert all(isolation["selections"].values())
    assert train["train_replay_ratio"] == pytest.approx(1.0)
    assert isinstance(train["train_replay_seed"], int)
    assert train["top_decks"]
    assert all(len(deck) == 60 for deck in train["top_decks"])
    assert train["ema_alpha"] == pytest.approx(0.99)
    assert config["model"]["norm_mode"] in {"prenorm", "postnorm"}


def test_submission_notebook_is_valid_json() -> None:
    notebook = json.loads(
        (
            PROJECT_ROOT / "kaggle_submission_imitation_agent.ipynb"
        ).read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "MODEL_PATH = Path(" in source
    assert "CG_PATH = Path(" in source
    assert "MODEL_CANDIDATES" not in source
    assert "CG_PATHS" not in source
    assert "ModelConfig(**checkpoint['config'])" in source
    assert "model.load_state_dict(checkpoint['model'])" in source


def test_deck_eda_notebook_starts_with_census_and_similarity() -> None:
    notebook = json.loads(
        (PROJECT_ROOT / "deck" / "deck_eda.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for heading in (
        "## Goal",
        "## Setup",
        "## Load & Validate",
        "## Deck Census",
        "## Pairwise Similarity",
        "## Checks",
        "## Top-Deck Archetype Isolation Roll",
        "## Deck Isolation Candidates",
        "## Deck Isolation Roll",
        "## Archetype Isolation Candidates",
        "## Archetype Isolation Roll",
        "## Combined Audit",
        "## Next Steps",
    ):
        assert heading in source
    assert "deck_summary.csv" in source
    assert "deck_similarity_pairs.csv" in source
    for text in (
        "DECK_MIN_REPLAYS = 100",
        "DECK_MIN_COUNT = 3",
        "DECK_TOTAL_REPLAYS_MIN = 1000",
        "DECK_TOTAL_REPLAYS_MAX = 3000",
        "DECK_ROLL_ID = 0",
        "ARCHETYPE_MIN_REPLAYS = 100",
        "ARCHETYPE_MIN_COUNT = 3",
        "ARCHETYPE_TOTAL_REPLAYS_MIN = 1000",
        "ARCHETYPE_TOTAL_REPLAYS_MAX = 3000",
        "ARCHETYPE_ROLL_ID = 0",
        "TOP_DECK_ARCHETYPE = 'Marnie Grimmsnarl'",
        "TOP_DECK_MIN_REPLAYS = 100",
        "TOP_DECK_MIN_COUNT = 3",
        "TOP_DECK_TOTAL_REPLAYS_MIN = 1000",
        "TOP_DECK_TOTAL_REPLAYS_MAX = 3000",
        "TOP_DECK_ROLL_ID = 0",
        "deck_isolation_selection.csv",
        "archetype_isolation_selection.csv",
        "top_deck_archetype_isolation_selection.csv",
    ):
        assert text in source
    assert "CARD_ISOLATION_ARCHETYPE_COUNT" not in source
    assert "isolation_recommendation.json" not in source
    assert "HIGH_SIM_MAX_CHANGED_SLOTS" not in source
    assert "MODERATE_SIM_MAX_CHANGED_SLOTS" not in source
    for text in (
        "reference_train_deck_id",
        "reference_changed_slots",
        "Automatic similarity cut points",
        "sampling_method_allowed",
        "build_top_deck_archetype_candidates",
        "roll_top_deck_archetype_selection",
        "excluded_archetypes=(TOP_DECK_ARCHETYPE,)",
        "PROJECT_ROOT / 'data' / 'deck_isolation_selection.csv'",
        "PROJECT_ROOT / 'data' / 'archetype_isolation_selection.csv'",
    ):
        assert text in source
    assert "PROJECT_ROOT / 'deck' / 'deck_isolation_selection.csv'" not in source
    assert (
        "PROJECT_ROOT / 'deck' / 'archetype_isolation_selection.csv'"
        not in source
    )
    top_deck_source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "TOP_DECK_CARD_IDS =" in "".join(cell.get("source", []))
    )
    tree = ast.parse(top_deck_source)
    top_deck_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "TOP_DECK_CARD_IDS"
            for target in node.targets
        )
    )
    top_deck_card_ids = ast.literal_eval(top_deck_assignment.value)
    assert len(top_deck_card_ids) == 60
    assert all(
        type(card_id) is int and card_id >= 0
        for card_id in top_deck_card_ids
    )
