from __future__ import annotations

from trellis.validate.closed_set import score_closed_set

# candidates = ["Jane Doe", "John Smith", "not mentioned"], not_mentioned_index=2
NOT_MENTIONED = 2


def test_present_correct() -> None:
    assert score_closed_set(chosen_index=0, correct_index=0, not_mentioned_index=NOT_MENTIONED) == (
        "present_correct"
    )


def test_present_incorrect() -> None:
    assert score_closed_set(chosen_index=1, correct_index=0, not_mentioned_index=NOT_MENTIONED) == (
        "present_incorrect"
    )


def test_silently_dropped() -> None:
    # correct value exists (index 0) but the model chose "not mentioned"
    assert score_closed_set(
        chosen_index=NOT_MENTIONED, correct_index=0, not_mentioned_index=NOT_MENTIONED
    ) == "silently_dropped"


def test_hallucinated() -> None:
    # field is genuinely absent (correct answer is "not mentioned") but the model chose a value
    assert score_closed_set(
        chosen_index=0, correct_index=NOT_MENTIONED, not_mentioned_index=NOT_MENTIONED
    ) == "hallucinated"


def test_correctly_absent() -> None:
    assert score_closed_set(
        chosen_index=NOT_MENTIONED, correct_index=NOT_MENTIONED, not_mentioned_index=NOT_MENTIONED
    ) == "correctly_absent"


def test_no_abstention_candidate_present_correct() -> None:
    # field spec has no "not mentioned" candidate at all
    assert score_closed_set(chosen_index=0, correct_index=0, not_mentioned_index=None) == (
        "present_correct"
    )


def test_no_abstention_candidate_present_incorrect() -> None:
    assert score_closed_set(chosen_index=1, correct_index=0, not_mentioned_index=None) == (
        "present_incorrect"
    )
