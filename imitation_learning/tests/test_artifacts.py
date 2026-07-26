from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_yaml_uses_validation_and_step_configuration() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "cfg" / "train.yaml").read_text(encoding="utf-8")
    )
    train = config["train"]
    assert "shuffle_mode" not in train
    assert "warmup_ratio" not in train
    assert train["warmup_steps"] >= 0
    assert train["validation_ratio"] == pytest.approx(0.05)
    assert train["ema_alpha"] == pytest.approx(0.99)
    assert config["model"]["norm_mode"] in {"prenorm", "postnorm"}


def test_submission_notebook_is_valid_json() -> None:
    notebook = json.loads(
        (
            PROJECT_ROOT / "kaggle_submission_imitation_agent.ipynb"
        ).read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
