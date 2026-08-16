from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "kaggle_submission_imitation_agent.ipynb"
)


def main_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "%%writefile main.py" in source:
            return source.removeprefix("%%writefile main.py\n")
    raise AssertionError("main.py cell not found")


def test_submission_maintains_known_deck_state_and_architecture() -> None:
    source = main_source()

    assert "ENCODER_TOKENS = 27" in source
    assert "'own_known_deck'" in source
    assert "known_deck_token_mlp_layers" in source
    assert "class KnownDeckTracker:" in source
    assert "KNOWN_DECK_TRACKER = KnownDeckTracker()" in source
    assert "KNOWN_DECK_TRACKER.reset()" in source
    assert "KNOWN_DECK_TRACKER.update(obs.logs, obs.current.yourIndex)" in source
    assert "known_deck_card_ids=KNOWN_DECK_TRACKER.card_ids()" in source
    compile(source, "generated-main.py", "exec")

