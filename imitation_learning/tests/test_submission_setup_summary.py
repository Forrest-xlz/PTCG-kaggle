from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "kaggle_submission_imitation_agent.ipynb"


def generated_main_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("%%writefile main.py"):
            return source.split("\n", 1)[1]
    raise AssertionError("main.py writefile cell not found")


def test_submission_contains_setup_summary_parity() -> None:
    source = generated_main_source()
    for expected in (
        "OWN_SUMMARY_DIM = 86",
        "OPPONENT_SUMMARY_DIM = 84",
        "class SetupFeatureCatalog:",
        "def build_setup_feature_catalog(",
        "def attack_energy_deficit(",
        "def public_setup_summary(",
        "def own_hand_setup_summary(",
        "value.energies",
        "active_physical_energy / 32.0",
        "active_effective_energy / 64.0",
        "own.extend(public_setup_summary(",
        "own.extend(own_hand_setup_summary(",
        "opponent.extend(public_setup_summary(",
    ):
        assert expected in source
    compile(source, "main.py", "exec")
