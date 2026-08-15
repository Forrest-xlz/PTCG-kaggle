from __future__ import annotations

import ast
import json
import sys
import unittest
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.revealed_hand import RevealedHandTracker
from model.attack_features import ATTACK_FEATURE_DIM
from model.card_features import CARD_FEATURE_DIM
from model.network import ModelConfig, PTCGTransformer
from model.features import damage_counter_action_eligibility

import torch


NOTEBOOK = PROJECT_ROOT / "kaggle_submission_imitation_agent.ipynb"


def _main_source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(payload["cells"][2]["source"])
    return source.removeprefix("%%writefile main.py\n")


def _config_source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(payload["cells"][1]["source"])


class _AreaType(IntEnum):
    BENCH = 5


def _counter_observation(context: int, hps: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        current=SimpleNamespace(
            players=[
                SimpleNamespace(bench=[]),
                SimpleNamespace(
                    bench=[SimpleNamespace(hp=hp) for hp in hps]
                ),
            ]
        ),
        select=SimpleNamespace(
            context=context,
            option=[
                SimpleNamespace(
                    area=_AreaType.BENCH,
                    playerIndex=1,
                    index=index,
                )
                for index in range(len(hps))
            ],
        ),
    )


def _notebook_counter_mask():
    tree = ast.parse(_main_source())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "damage_counter_action_eligibility"
    )
    namespace = {"np": __import__("numpy"), "AreaType": _AreaType}
    exec(compile(ast.Module([function], []), str(NOTEBOOK), "exec"), namespace)
    return namespace["damage_counter_action_eligibility"]


def _notebook_tracker_class():
    tree = ast.parse(_main_source())
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "optional_log_int"
        )
        or (
            isinstance(node, ast.ClassDef)
            and node.name == "RevealedHandTracker"
        )
    ]
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(selected, []), str(NOTEBOOK), "exec"), namespace)
    return namespace["RevealedHandTracker"]


def _notebook_model_namespace() -> dict[str, object]:
    tree = ast.parse(_main_source())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SparseVector":
            break
        if isinstance(node, ast.ImportFrom) and node.module == "cg.api":
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MY_DECK"
            for target in node.targets
        ):
            continue
        selected.append(node)
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(selected, []), str(NOTEBOOK), "exec"), namespace)
    return namespace


def _log(log_type, player=0, card=None, serial=None, source=None, target=None):
    return SimpleNamespace(
        type=log_type,
        playerIndex=player,
        cardId=card,
        serial=serial,
        fromArea=source,
        toArea=target,
    )


class SubmissionNotebookTests(unittest.TestCase):
    def test_generated_agent_source_compiles(self) -> None:
        compile(_main_source(), str(NOTEBOOK), "exec")

    def test_notebook_tracker_matches_training_tracker(self) -> None:
        notebook_tracker = _notebook_tracker_class()()
        project_tracker = RevealedHandTracker()
        logs = [
            _log(6, 0, 741, 10, 12, 2),
            _log(6, 1, 305, 20, 12, 2),
            _log(4, 1, 66, 21),
            _log(10, 0, 741, 10),
        ]

        notebook_tracker.update(logs)
        project_tracker.update(logs)

        self.assertEqual(
            notebook_tracker.relative_cards(0),
            project_tracker.relative_cards(0),
        )

    def test_damage_counter_mask_matches_project_behavior(self) -> None:
        api = ModuleType("cg.api")
        api.AreaType = _AreaType
        cg = ModuleType("cg")
        cg.api = api
        previous_cg = sys.modules.get("cg")
        previous_api = sys.modules.get("cg.api")
        sys.modules["cg"] = cg
        sys.modules["cg.api"] = api
        try:
            notebook_mask = _notebook_counter_mask()
            for context, hps, actions in (
                (14, [0, 40], [[0], [1]]),
                (14, [0, -10], [[0], [1]]),
                (14, [0, 40], [[0, 1], [1]]),
                (13, [0, 40], [[0], [1]]),
            ):
                obs = _counter_observation(context, hps)
                expected = damage_counter_action_eligibility(obs, actions)
                actual = notebook_mask(obs, actions)
                self.assertEqual(actual.tolist(), expected.tolist())
        finally:
            if previous_cg is None:
                sys.modules.pop("cg", None)
            else:
                sys.modules["cg"] = previous_cg
            if previous_api is None:
                sys.modules.pop("cg.api", None)
            else:
                sys.modules["cg.api"] = previous_api

    def test_config_cell_exposes_boolean_damage_counter_switch(self) -> None:
        tree = ast.parse(_config_source())
        assignments = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id == "DAMAGE_COUNTER_KO_MASK"
        }
        self.assertIn("DAMAGE_COUNTER_KO_MASK", assignments)
        self.assertIs(type(assignments["DAMAGE_COUNTER_KO_MASK"]), bool)

    def test_notebook_model_state_shapes_match_project_model(self) -> None:
        namespace = _notebook_model_namespace()
        kwargs = dict(
            card_count=10,
            attack_count=5,
            encoder_size=256,
            d_model=8,
            num_heads=2,
            d_feedforward=16,
            encoder_layers=1,
            decoder_layers=1,
            revealed_hand_token_mlp_layers=1,
            learnable_cls_token=True,
        )
        card_features = torch.zeros(10, CARD_FEATURE_DIM)
        attack_features = torch.zeros(5, ATTACK_FEATURE_DIM)
        project_model = PTCGTransformer(
            ModelConfig(**kwargs), card_features, attack_features
        )
        notebook_model = namespace["PTCGTransformer"](
            namespace["ModelConfig"](**kwargs),
            card_features,
            attack_features,
        )

        project_shapes = {
            name: tuple(value.shape)
            for name, value in project_model.state_dict().items()
        }
        notebook_shapes = {
            name: tuple(value.shape)
            for name, value in notebook_model.state_dict().items()
        }
        self.assertEqual(notebook_shapes, project_shapes)
        notebook_model.load_state_dict(project_model.state_dict(), strict=True)
        project_model.eval()
        notebook_model.eval()

        revealed_base = 8 + 17 * kwargs["card_count"]
        categorical = torch.tensor(
            [[0, 0, kwargs["card_count"], kwargs["card_count"],
              kwargs["attack_count"], 0, 0, 0, 0, 0, 0]],
            dtype=torch.long,
        )
        inputs = (
            torch.tensor([revealed_base + 7]),
            torch.tensor([1.0]),
            torch.tensor([0] * 27 + [1]),
            torch.zeros(1, 18, dtype=torch.long),
            torch.zeros(1, 69),
            torch.zeros(1, 71),
            torch.zeros(1, 73),
            torch.tensor([[1, 0]]),
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
            torch.empty(0, 11, dtype=torch.long),
            torch.empty(0, 8, dtype=torch.long),
            torch.empty(0, 46),
            torch.empty(0, 6),
            torch.zeros(4, dtype=torch.long),
            categorical,
            torch.zeros(1, 5),
            torch.zeros(1, 46),
            torch.zeros(1, 6),
            torch.tensor([0]),
            torch.tensor([0, 1]),
        )
        with torch.inference_mode():
            project_logits = project_model(*inputs)
            notebook_logits = notebook_model(*inputs)
        torch.testing.assert_close(notebook_logits, project_logits)


if __name__ == "__main__":
    unittest.main()
