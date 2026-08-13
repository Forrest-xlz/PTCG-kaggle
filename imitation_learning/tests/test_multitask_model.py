import torch

from model.attack_features import ATTACK_FEATURE_DIM
from model.card_features import CARD_FEATURE_DIM
from model.network import ModelConfig, PTCGTransformer


def make_model(**overrides) -> PTCGTransformer:
    config = ModelConfig(
        card_count=8,
        attack_count=4,
        d_model=8,
        num_heads=2,
        d_feedforward=16,
        **overrides,
    )
    return PTCGTransformer(
        config,
        torch.zeros(config.card_count, CARD_FEATURE_DIM),
        torch.zeros(config.attack_count, ATTACK_FEATURE_DIM),
    )


def test_auxiliary_off_keeps_baseline_encoder_shape_and_no_heads() -> None:
    model = make_model()
    assert model.encoder_token_count == 26
    assert model.cls_token is None
    assert model.next_select_type_head is None


def test_cls_state_tasks_prepend_one_unmasked_token() -> None:
    model = make_model(
        auxiliary_state_representation="cls",
        opponent_deck_auxiliary=True,
        opponent_deck_class_count=5,
    )
    mask = torch.tensor([[False, True]])
    assert model.encoder_token_count == 27
    assert model._prepend_cls_mask(mask).tolist() == [[False, False, True]]


def test_auxiliary_heads_use_selected_action_and_state_token() -> None:
    model = make_model(
        next_decision_auxiliary=True,
        auxiliary_state_representation="global",
        opponent_deck_auxiliary=True,
        opponent_deck_class_count=5,
        final_own_prize_auxiliary=True,
    )
    decoded_actions = torch.randn(2, 4, 8)
    state = torch.randn(2, 8)
    outputs = model.auxiliary_logits(
        decoded_actions, state, torch.tensor([1, 3])
    )

    assert outputs["next_select_type"].shape == (2, 11)
    assert outputs["next_select_context"].shape == (2, 49)
    assert outputs["opponent_deck"].shape == (2, 5)
    assert outputs["final_own_prize"].shape == (2, 7)


def test_invalid_auxiliary_configuration_is_rejected() -> None:
    try:
        make_model(opponent_deck_auxiliary=True, opponent_deck_class_count=0)
    except ValueError as exc:
        assert "opponent_deck_class_count" in str(exc)
    else:
        raise AssertionError("enabled opponent Deck task requires classes")
