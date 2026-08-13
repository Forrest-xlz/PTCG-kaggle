from training.extract import _annotate_auxiliary_labels, _final_own_prize_counts


def observation(your_index: int, own_prizes: int, select_type: int, context: int):
    players = [{"prize": [None] * 6}, {"prize": [None] * 6}]
    players[your_index]["prize"] = [None] * own_prizes
    return {
        "current": {"yourIndex": your_index, "players": players},
        "select": {"type": select_type, "context": context},
    }


def test_auxiliary_labels_follow_same_player_and_terminal_own_prizes() -> None:
    decks = [[1] * 60, [2] * 60]
    records = {
        0: [
            {"observation": observation(0, 5, 1, 10)},
            {"observation": observation(0, 2, 3, 12)},
        ],
        1: [{"observation": observation(1, 4, 2, 11)}],
    }

    _annotate_auxiliary_labels(records, decks, [2, 4])

    assert records[0][0]["next_decision_valid"] is True
    assert records[0][0]["next_select_type"] == 3
    assert records[0][0]["next_select_context"] == 12
    assert records[0][1]["next_decision_valid"] is False
    assert records[0][0]["final_own_prize_count"] == 2
    assert records[1][0]["final_own_prize_count"] == 4
    assert records[0][0]["opponent_deck_id"] != records[1][0]["opponent_deck_id"]


def test_final_prize_count_uses_last_available_observation_not_last_decision() -> None:
    steps = [
        [
            {"observation": observation(0, 5, 1, 10)},
            {"observation": observation(1, 5, 2, 11)},
        ],
        [
            {"observation": observation(0, 1, 0, 0)},
            {"observation": observation(1, 3, 0, 0)},
        ],
    ]

    assert _final_own_prize_counts(steps, 2) == [1, 3]
