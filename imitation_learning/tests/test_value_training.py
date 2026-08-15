from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.feature_cache import EncoderBatch, PLAYER_RESULT_WIN
from training.precision import PrecisionContext
from value.train import (
    build_lr_scheduler,
    load_value_settings,
    load_settings,
    should_trigger,
    train_value_batch,
    validation_namespaces,
    value_checkpoint_payload,
)


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "value.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version_name": "value_test",
                "value_train": {
                    "cg_path": "../engine",
                    "data": "data/cache",
                    "replay_episodes": "../replays",
                    "pretrained_checkpoint": "outputs/policy.pt",
                    "output": "outputs/${version_name}",
                    "resume": False,
                    "resume_checkpoint": None,
                    "epochs": 2,
                    "batch_size": 4,
                    "learning_rate": 0.001,
                    "weight_decay": 0.01,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "warmup_steps": 1,
                    "eval_every_steps": 2,
                    "save_every_steps": 3,
                    "save_every_epoch": True,
                    "validation_ratio": 0.05,
                    "validation_seed": 42,
                    "expert_validation_ratio": 0.05,
                    "seed": 7,
                    "device": "cpu",
                    "precision": "fp32",
                    "log_every_steps": 1,
                    "grad_clip_norm": 1.0,
                    "ema_alpha": 0.99,
                    "isolation_validation": {
                        "deck_data": "data/deck",
                        "selections": {
                            "deck_isolation": "data/a.csv",
                            "archetype_isolation": "data/b.csv",
                            "top_deck_archetype_isolation": "data/c.csv",
                        },
                    },
                    "top_decks": [list(range(60))],
                },
                "value_model": {
                    "head_layers": 2,
                    "dropout": 0.1,
                    "output_activation": "tanh",
                },
                "wandb": {
                    "enabled": False,
                    "project": "ptcg_value",
                    "group": "value",
                    "name": "${version_name}",
                    "mode": "offline",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_load_settings_expands_version_and_keeps_architecture_out_of_yaml(
    tmp_path: Path,
) -> None:
    settings = load_settings(_config(tmp_path))
    assert settings.version_name == "value_test"
    assert settings.train.output == "outputs/value_test"
    assert settings.wandb.name == "value_test"
    assert settings.model.head_layers == 2
    assert settings.train.pretrained_checkpoint == "outputs/policy.pt"
    assert load_value_settings(_config(tmp_path)) == settings


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("value_model", "head_layers", 0, "head_layers"),
        ("value_model", "dropout", 1.0, "dropout"),
        ("value_model", "output_activation", "linear", "tanh"),
        ("value_train", "eval_every_steps", 0, "eval_every_steps"),
    ],
)
def test_load_settings_rejects_invalid_values(
    tmp_path: Path, section: str, key: str, value, message: str
) -> None:
    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text("utf-8"))
    raw[section][key] = value
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_settings(path)


def _batch() -> EncoderBatch:
    return EncoderBatch(
        encoder_index=np.zeros(52, dtype=np.int32),
        encoder_value=np.ones(52, dtype=np.float16),
        encoder_offset=np.arange(52, dtype=np.int32),
        encoder_pokemon_appear=np.zeros((2, 18), dtype=np.uint8),
        own_summary=np.pad(np.array([[0.0], [0.5]]), ((0, 0), (0, 68))).astype(np.float16),
        opponent_summary=np.zeros((2, 71), dtype=np.float16),
        global_summary=np.zeros((2, 73), dtype=np.float16),
        history_select_type=np.zeros((2, 3), dtype=np.uint8),
        history_select_context=np.zeros((2, 3), dtype=np.uint8),
        history_valid=np.zeros((2, 3), dtype=np.uint8),
        history_option_categorical=np.empty((0, 11), dtype=np.int64),
        history_structural=np.empty((0, 8), dtype=np.int64),
        history_pokemon_dynamic=np.empty((0, 46), dtype=np.float16),
        history_attack_dynamic=np.empty((0, 6), dtype=np.float16),
        history_option_offset=np.zeros(7, dtype=np.int32),
        player_result=np.array([PLAYER_RESULT_WIN, PLAYER_RESULT_WIN], dtype=np.uint8),
    )


class TinyValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, *inputs):
        return torch.tanh(inputs[4][:, 0] * self.weight)


def test_train_value_batch_updates_model_once() -> None:
    model = TinyValue()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = build_lr_scheduler(optimizer, total_steps=2, warmup_steps=0)
    result = train_value_batch(
        batch=_batch(),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        precision=PrecisionContext("fp32", torch.device("cpu")),
        grad_clip_norm=1.0,
    )
    assert result.samples == 2
    assert result.mse == pytest.approx(1.0)
    assert model.weight.item() > 0.0


def test_checkpoint_and_namespaces_include_value_metadata() -> None:
    model = TinyValue()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = build_lr_scheduler(optimizer, total_steps=2, warmup_steps=0)
    precision = PrecisionContext("fp32", torch.device("cpu"))
    payload = value_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        precision=precision,
        encoder_config={"d_model": 8},
        value_model_config={"head_layers": 2},
        global_step=3,
        epoch=1,
        ema={"mse": {"value": 0.5}},
        history=[],
    )
    assert payload["encoder_config"]["d_model"] == 8
    assert payload["global_step"] == 3
    namespaces = validation_namespaces(
        2,
        ("deck_isolation", "archetype_isolation"),
    )
    assert "val_latest_expert_deck2" in namespaces
    assert "val_in_distribution_deck1" in namespaces
    assert "val_deck_isolation" in namespaces
    assert "deck_isolation" not in namespaces
    assert should_trigger(5, 10)
    assert not should_trigger(5, 0)
