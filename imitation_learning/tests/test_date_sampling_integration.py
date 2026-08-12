from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("pandas")

from training.train import (
    effective_epoch_samples,
    load_settings,
    prepare_date_sampled_training_indices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "cfg" / "train.yaml").read_text(encoding="utf-8")
    )


def write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_date_sampling_settings_parse(tmp_path: Path) -> None:
    value = config()
    value["train"]["date_sampling"] = {
        "enabled": True,
        "mode": "power",
        "seed": 17,
        "linear": {"start": 0.5, "end": 1.2},
        "power": {"start": 0.4, "end": 1.5, "exponent": 2.0},
    }

    settings = load_settings(write_config(tmp_path, value)).train.date_sampling

    assert settings.enabled is True
    assert settings.mode == "power"
    assert settings.seed == 17
    assert settings.power.exponent == 2.0


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("mode",), "other", "mode"),
        (("seed",), 1.5, "seed"),
        (("linear", "start"), -0.1, "linear.start"),
        (("power", "end"), float("inf"), "power.end"),
        (("power", "exponent"), 0.0, "power.exponent"),
    ],
)
def test_date_sampling_settings_reject_invalid_values(
    tmp_path: Path, path: tuple[str, ...], value, message: str
) -> None:
    raw = config()
    current = raw["train"]["date_sampling"]
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        load_settings(write_config(tmp_path, raw))


class FakeDataset:
    def dates_for_indices(self, indices: np.ndarray) -> np.ndarray:
        assert indices.tolist() == [10, 11, 12, 13]
        return np.asarray([[7, 1], [7, 1], [7, 5], [7, 5]], dtype=np.int16)


def test_training_indices_are_weighted_after_receiving_final_split(
    tmp_path: Path,
) -> None:
    raw = config()
    raw["train"]["date_sampling"] = {
        "enabled": True,
        "mode": "linear",
        "seed": 3,
        "linear": {"start": 1.0, "end": 2.0},
        "power": {"start": 1.0, "end": 2.0, "exponent": 2.0},
    }
    settings = load_settings(write_config(tmp_path, raw)).train.date_sampling
    original = np.asarray([10, 11, 12, 13], dtype=np.uint32)

    result = prepare_date_sampled_training_indices(
        FakeDataset(), original, settings
    )

    assert len(result.indices) == 6
    assert np.count_nonzero(result.indices == 10) == 1
    assert np.count_nonzero(result.indices == 12) == 2
    assert effective_epoch_samples(result.indices, None) == 6
    assert effective_epoch_samples(result.indices, 5) == 5
