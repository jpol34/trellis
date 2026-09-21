"""Scoring for trellis/Jev: both are closed-set arms that pick a candidate index rather than
generate free text, so correctness reduces to index comparison."""

from __future__ import annotations

from trellis.validate.states import FiveState, resolve_state


def score_closed_set(
    chosen_index: int,
    correct_index: int,
    not_mentioned_index: int | None,
) -> FiveState:
    """Scores one closed-set prediction. `not_mentioned_index` is the candidate index reserved
    for "not mentioned in transcript"; passing `None` means the field had no such candidate
    (every candidate is a real value, so abstention isn't representable)."""
    field_required_and_stated_in_gold = correct_index != not_mentioned_index
    model_abstained = chosen_index == not_mentioned_index

    if model_abstained:
        model_correct = None
    else:
        model_correct = chosen_index == correct_index

    return resolve_state(field_required_and_stated_in_gold, model_abstained, model_correct)
