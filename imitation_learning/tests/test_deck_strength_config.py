from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deck_strength.config import load_settings, schedule_games


def _write_config(tmp_path: Path, *, games: int = 3) -> Path:
    path = tmp_path / "deck_strength.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "deck_strength": {
                    "cg_path": "../engine",
                    "checkpoint": "outputs/policy.pt",
                    "device": "cpu",
                    "seed": 17,
                    "games_per_pair": games,
                    "output": "outputs/strength-test",
                    "runtime": {
                        "workers": 2,
                        "torch_threads_per_worker": 1,
                    },
                    "decks": [
                        {"name": "a", "cards": list(range(60))},
                        {"name": "b", "cards": list(reversed(range(60)))},
                        {"name": "c", "cards": [3] * 60},
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_load_settings_resolves_paths_and_runtime(tmp_path: Path) -> None:
    settings = load_settings(_write_config(tmp_path))

    assert settings.cg_path == (PROJECT_ROOT / "../engine").resolve()
    assert settings.checkpoint == PROJECT_ROOT / "outputs/policy.pt"
    assert settings.output == PROJECT_ROOT / "outputs/strength-test"
    assert settings.device == "cpu"
    assert settings.seed == 17
    assert settings.games_per_pair == 3
    assert settings.runtime.workers == 2
    assert settings.runtime.torch_threads_per_worker == 1
    assert [deck.name for deck in settings.decks] == ["a", "b", "c"]


def test_schedule_uses_each_unordered_pair_once_and_balances_seats(
    tmp_path: Path,
) -> None:
    settings = load_settings(_write_config(tmp_path, games=3))
    games = schedule_games(settings)

    assert len(games) == 9
    assert {(game.deck_a.name, game.deck_b.name) for game in games} == {
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    }
    assert all(game.deck_a != game.deck_b for game in games)
    for pair in (("a", "b"), ("a", "c"), ("b", "c")):
        cell = [
            game
            for game in games
            if (game.deck_a.name, game.deck_b.name) == pair
        ]
        assert len(cell) == 3
        seats = [sum(game.deck_a_player == seat for game in cell) for seat in (0, 1)]
        assert abs(seats[0] - seats[1]) <= 1
    assert [game.game_id for game in games] == list(range(9))
    assert len({game.seed for game in games}) == 9


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("games_per_pair",), 0, "games_per_pair"),
        (("runtime", "workers"), 0, "runtime.workers"),
        (("runtime", "torch_threads_per_worker"), False, "torch_threads"),
        (("decks", 0, "cards"), [1, 2], "60"),
        (("decks", 1, "name"), "a", "unique"),
    ],
)
def test_load_settings_rejects_invalid_values(
    tmp_path: Path,
    path: tuple[object, ...],
    value,
    message: str,
) -> None:
    config_path = _write_config(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target = raw["deck_strength"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_settings(config_path)
