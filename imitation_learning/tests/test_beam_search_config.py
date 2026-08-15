from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beam_search.config import load_settings, schedule_games


def _write_config(tmp_path: Path, *, games: int = 3) -> Path:
    path = tmp_path / "beam_search.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "beam_search": {
                    "cg_path": "../engine",
                    "checkpoint": "outputs/policy.pt",
                    "device": "cpu",
                    "seed": 17,
                    "games_per_matchup": games,
                    "output": "outputs/beam-test",
                    "runtime": {
                        "workers": 2,
                        "torch_threads_per_worker": 1,
                    },
                    "search": {
                        "beam_width": 4,
                        "expansion_top_k": 3,
                        "alpha": 0.8,
                        "max_depth": 40,
                    },
                    "decks": [
                        {"name": "a", "cards": list(range(60))},
                        {"name": "b", "cards": list(reversed(range(60)))},
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_load_settings_resolves_paths_and_values(tmp_path: Path) -> None:
    settings = load_settings(_write_config(tmp_path))

    assert settings.cg_path == (PROJECT_ROOT / "../engine").resolve()
    assert settings.checkpoint == PROJECT_ROOT / "outputs/policy.pt"
    assert settings.output == PROJECT_ROOT / "outputs/beam-test"
    assert settings.device == "cpu"
    assert settings.seed == 17
    assert settings.games_per_matchup == 3
    assert settings.runtime.workers == 2
    assert settings.runtime.torch_threads_per_worker == 1
    assert settings.search.beam_width == 4
    assert settings.search.expansion_top_k == 3
    assert settings.search.alpha == pytest.approx(0.8)
    assert settings.search.max_depth == 40
    assert [deck.name for deck in settings.decks] == ["a", "b"]


def test_schedule_contains_every_ordered_cell_and_balances_seats(
    tmp_path: Path,
) -> None:
    settings = load_settings(_write_config(tmp_path, games=3))
    scheduled = schedule_games(settings)

    assert len(scheduled) == 12
    assert {
        (game.beam_deck.name, game.greedy_deck.name)
        for game in scheduled
    } == {("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")}
    for beam_name in ("a", "b"):
        for greedy_name in ("a", "b"):
            cell = [
                game
                for game in scheduled
                if game.beam_deck.name == beam_name
                and game.greedy_deck.name == greedy_name
            ]
            assert len(cell) == 3
            seats = [sum(game.beam_player == seat for game in cell) for seat in (0, 1)]
            assert abs(seats[0] - seats[1]) <= 1
    assert [game.game_id for game in scheduled] == list(range(12))
    assert len({game.seed for game in scheduled}) == 12


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("games_per_matchup",), 0, "games_per_matchup"),
        (("runtime", "workers"), 0, "runtime.workers"),
        (
            ("runtime", "torch_threads_per_worker"),
            False,
            "torch_threads_per_worker",
        ),
        (("search", "beam_width"), 0, "beam_width"),
        (("search", "expansion_top_k"), False, "expansion_top_k"),
        (("search", "alpha"), -0.1, "alpha"),
        (("search", "max_depth"), 0, "max_depth"),
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
    target = raw["beam_search"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        load_settings(config_path)
