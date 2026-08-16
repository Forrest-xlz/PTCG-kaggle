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


def test_train_yaml_uses_full_data_and_step_configuration() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "cfg" / "train.yaml").read_text(encoding="utf-8")
    )
    train = config["train"]
    assert train["resume"] is False
    assert train["resume_checkpoint"] is None
    assert "shuffle_mode" not in train
    assert "warmup_ratio" not in train
    assert train["warmup_steps"] >= 0
    validation_only = {
        "eval_every_steps",
        "validation_ratio",
        "validation_seed",
        "expert_validation_ratio",
        "isolation_validation",
        "top_decks",
    }
    assert validation_only.isdisjoint(train)
    loser_augmentation = train["loser_augmentation"]
    assert loser_augmentation["enabled"] is True
    assert loser_augmentation["recent_dates"] >= 1
    assert 0 < loser_augmentation["expert_ratio"] <= 1
    assert train["replay_episodes"]
    assert train["train_replay_ratio"] == pytest.approx(1.0)
    assert isinstance(train["train_replay_seed"], int)
    assert train["ema_alpha"] == pytest.approx(0.99)
    assert config["model"]["norm_mode"] in {"prenorm", "postnorm"}
    assert config["model"]["card_mlp_layers"] >= 0
    assert config["model"]["card_mlp_scope"] in {"shared", "region"}
    assert config["model"]["pokemon_appear_embedding"] is True
    for name in (
        "bench_token_mlp_layers",
        "active_token_mlp_layers",
        "discard_token_mlp_layers",
        "hand_token_mlp_layers",
        "deck_token_mlp_layers",
        "known_deck_token_mlp_layers",
    ):
        assert config["model"][name] >= 0
    assert isinstance(config["model"]["region_token_mlp_residual"], bool)


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
    assert "card_mlp_layers" in source
    assert "card_mlp_scope" in source
    assert "card_feature_projections" in source
    assert "index_to_card_region" in source
    assert "pokemon_appear_embedding" in source
    assert "pokemon_appear" in source
    assert "apply_region_token_mlps" in source
    assert "own_bench_token_mlp" in source
    assert "opponent_bench_token_mlp" in source
    assert "region_token_mlp_residual" in source
    assert "build_card_feature_table" in source
    assert "build_attack_feature_table" in source
    assert "projected_card_features" in source
    assert "option_categorical" in source
    assert "option_numeric" in source
    assert "OPTION_NUMERIC_DIM = 76" in source
    assert "set_one_hot" in source
    assert "option_candidate_embedding" in source
    assert "option_target_embedding" in source
    assert "option_attack_embedding" in source
    assert "include_last_offset=True" in source
    assert "decoder_bag" not in source
    assert "ENCODER_TOKENS = 26" in source
    assert "num_encoder_words" not in source
    assert "own_summary_projection" in source
    assert "opponent_summary_projection" in source
    assert "global_summary_projection" in source
    assert "SELECT_TYPE_DIM = 11" in source
    assert "SELECT_CONTEXT_DIM = 49" in source
    assert "for slot in range(8)" in source
    assert "players[0].discard, 0.25" in source
    assert "players[1].discard, 0.25" in source


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
